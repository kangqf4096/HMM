"""Offline vLLM 0.8.4 V0 generation of L0 sources (Linux/CUDA only; no edits, no replay)."""
import os
from pathlib import Path
from time import perf_counter

from ..data import DATASET, REVISION, SUBSET
from ..io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path
from ..progress import progress, tick
from ..protocol import PROTOCOL, PUBLIC_KEY, context_seed, generation_contexts, pivot
from .config import validate_l0
from .schema import scenario_for, scenario_ranges

os.environ.setdefault("VLLM_USE_V1", "0")


def plan_jobs(prompts, config):
    """One generation job per non-pool article; scenarios are fixed by role order."""
    seen, jobs = {}, []
    for doc in prompts:
        role = doc["role"]
        if role == "pool":
            continue
        index = seen.get(role, 0)
        seen[role] = index + 1
        scenario = scenario_for(role, index)
        ranges = scenario_ranges(scenario, config)
        jobs.append({"doc": doc, "scenario": scenario, "ranges": ranges})
    return jobs


def source_record(job, tokens, audit, key, context_width):
    """Verify the sampler audit against the emitted tokens and build the source row."""
    doc, ranges = job["doc"], job["ranges"]
    contexts = generation_contexts(doc["prompt_ids"], tokens, context_width)
    seeds = [str(context_seed(c, key)) for c in contexts]
    enabled = [any(a <= t < b for a, b in ranges) for t in range(len(tokens))]
    expected = set(range(len(tokens)))
    missing, extra = sorted(expected - set(audit)), sorted(set(audit) - expected)
    # vLLM V0 calls the processor once more after the last emitted token; that draw is discarded.
    if missing or set(extra) - {len(tokens)}:
        raise RuntimeError(f"{doc['group']}: generation audit has {len(audit)} steps; missing {missing[:10]}, "
                           f"unexpected {extra[:10]}; logits processor did not run as expected.")
    audit = {t: row for t, row in audit.items() if t < len(tokens)}
    for t, row in audit.items():
        if row["seed"] != seeds[t] or row["context"] != contexts[t] or row["active"] != enabled[t]:
            raise RuntimeError("Generation/verification context or seed mismatch.")
        if enabled[t] and (row["chosen"] != tokens[t] or row["selected_y"] != pivot(int(seeds[t]), tokens[t])):
            raise RuntimeError("Watermark sampler audit does not match emitted tokens.")
    return {"article_id": job.get("article_id", doc["group"]), "split": doc["role"], "scenario": job["scenario"],
            "ranges": ranges, "prompt_ids": doc["prompt_ids"], "token_ids": [int(t) for t in tokens],
            "source_split": doc.get("source_split"), "source_row": doc.get("source_row"),
            **job.get("extra", {})}


def generate(config, model, prompts_dir, output, tensor_parallel=1, gpu_memory=.8, dtype="float16", *,
             validate=validate_l0, planner=plan_jobs, key=None, key_id=None, chunk=None, tick_stage=None):
    model, prompts_dir, output = (project_path(p) for p in (model, prompts_dir, output))
    model = str(model)
    validate(config)
    if tensor_parallel < 1 or not 0 < gpu_memory <= 1:
        raise ValueError("Invalid tensor parallelism or GPU memory fraction.")
    if not Path(model).is_dir():
        raise ValueError(f"Local model directory does not exist: {model}")
    manifest = read_json(Path(prompts_dir) / "manifest.json")
    for name, default in (("dataset", DATASET), ("subset", SUBSET), ("dataset_revision", REVISION)):
        if manifest.get(name) != config.get(name, default):
            raise ValueError(f"Preparation and generation differ on {name}. Prepare again.")
    for name in ("context_width", "prompt_tokens", "roles"):
        if manifest["config"].get(name) != config.get(name):
            raise ValueError(f"Preparation and generation differ on {name}. Prepare again.")
    if Path(manifest["model"]).resolve() != Path(model).resolve():
        source_path = Path(model) / "opt_hmm_source.json"
        source = read_json(source_path) if source_path.is_file() else {}
        prepared = manifest.get("model_source") or {}
        if not all(source.get(k) and source[k] == prepared.get(k) for k in ("repo_id", "revision")):
            raise ValueError("Tokenized prompts belong to a different model. Prepare again with the new model.")
    from ..runtime import check_runtime
    from ..vllm_processor import GumbelRequest, validate_sampling_params
    progress("检查 GPU 运行环境")
    runtime = check_runtime(model)
    output = new_output(output)
    trace_dir = output / "trace"
    trace_dir.mkdir()
    jobs = planner(read_jsonl(Path(prompts_dir) / "prompts.jsonl"), config)
    progress("加载 vLLM")
    from vllm import LLM, SamplingParams
    key = key or os.environ.get("OPT_HMM_KEY", PUBLIC_KEY)
    n, m, temperature = config["tokens"], config["context_width"], config.get("temperature", 1.)
    seed = config.get("seed", 20261015)
    started = perf_counter()
    scheduler_options = {}
    if "max_num_batched_tokens" in config:
        scheduler_options["max_num_batched_tokens"] = config["max_num_batched_tokens"]
    llm = LLM(model=model, tokenizer=model, dtype=dtype, tensor_parallel_size=tensor_parallel,
              gpu_memory_utilization=gpu_memory, max_model_len=config["max_model_len"],
              max_num_seqs=config["max_num_seqs"], seed=seed, enforce_eager=config.get("enforce_eager", False),
              enable_prefix_caching=False, generation_config="vllm", **scheduler_options)
    model_load_seconds = perf_counter() - started
    tokenizer = llm.get_tokenizer()
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    valid_ids = set(tokenizer.get_vocab().values())
    if max(valid_ids) >= vocab_size:
        raise ValueError("Tokenizer has added IDs outside model output vocabulary; reconcile model/tokenizer.")
    blocked = sorted((set(range(vocab_size)) - valid_ids) | set(tokenizer.all_special_ids))
    storage = config.get("generation_storage")
    if storage:
        article_count, generation_seconds = _cached_sources(
            jobs, llm, SamplingParams, GumbelRequest, validate_sampling_params,
            config, key, vocab_size, blocked, manifest, output,
            chunk or len(jobs), tick_stage)
    else:
        params, requests = [], []
        for number, job in enumerate(jobs):
            settings = {"context_width": m, "key": key, "vocab_size": vocab_size, "blocked_ids": blocked,
                        "ranges": job["ranges"], "trace_path": str((trace_dir / f"r{number}.jsonl").resolve()),
                        "pivot_diagnostics": False}
            job["trace_path"] = settings["trace_path"]
            params.append(SamplingParams(n=1, temperature=temperature, top_p=1., top_k=-1, min_p=0.,
                                         max_tokens=n, ignore_eos=True, detokenize=False, seed=seed + number,
                                         logits_processors=[GumbelRequest(settings, temperature)]))
            validate_sampling_params(params[-1])
            requests.append({"prompt_token_ids": job["doc"]["prompt_ids"]})
        progress("生成实验文本", 0, len(jobs), "生成请求进度由 vLLM 报告")
        started = perf_counter()
        size = chunk or len(requests)
        outputs = []
        for start in range(0, len(requests), size):
            part = llm.generate(requests[start:start + size], sampling_params=params[start:start + size], use_tqdm=True)
            outputs += part
            if tick_stage:
                tick(tick_stage, len(part))
        generation_seconds = perf_counter() - started
        if len(outputs) != len(jobs):
            raise RuntimeError("vLLM output count mismatch.")
        sources = []
        for number, (job, generated) in enumerate(zip(jobs, outputs), 1):
            if list(generated.prompt_token_ids) != job["doc"]["prompt_ids"]:
                raise RuntimeError("vLLM request ordering/prompt mismatch.")
            tokens = list(generated.outputs[0].token_ids)
            if len(tokens) != n:
                raise RuntimeError(f"Incomplete generation for {job['doc']['group']}: {len(tokens)} != {n}")
            audit = {row["step"]: row for row in read_jsonl(job["trace_path"])}
            sources.append(source_record(job, tokens, audit, key, m))
            progress("核验并整理生成结果", number, len(jobs))
        write_jsonl(output / "sources.jsonl", sources)
        article_count = len(sources)
    write_json(output / "metadata.json", {
        "config": config, "model": model, "runtime": runtime, "provenance": provenance(),
        "protocol": PROTOCOL, "key_id": key_id or config.get("key_id", "public-research-v1"),
        # Research phase: the plaintext key keeps a run reproducible; see docs/reproducibility_and_secrets_20261101.md.
        "key": key, "public_research_key": key == PUBLIC_KEY, "vocab_size": vocab_size,
        "blocked_ids": blocked, "tensor_parallel": tensor_parallel, "dtype": dtype,
        "gpu_memory_utilization": gpu_memory, "enforce_eager": config.get("enforce_eager", False),
        "model_load_seconds": model_load_seconds, "generation_seconds": generation_seconds,
        "audit": "Every active chosen token, pivot, context and seed verified.", "prompt_manifest": manifest})
    return {"articles": article_count, "generation_seconds": generation_seconds}


def _cached_sources(jobs, llm, SamplingParams, GumbelRequest, validate_params,
                    config, key, vocab_size, blocked, manifest, output, size, tick_stage):
    import shutil
    from ..batch_cache import BatchCache

    storage = config["generation_storage"]
    cache = BatchCache(project_path(storage["cache_root"]),
                       int(storage.get("reserve_gb", 1) * 1024 ** 3))
    n, m = config["tokens"], config["context_width"]
    temperature, seed = config.get("temperature", 1.), config.get("seed", 20261015)
    # Conservative allowance for trace plus inputs and verified source records.
    budget = max(1024 ** 2, n * 4096)
    started = perf_counter()
    count = 0
    progress("本地双批缓存", detail=str(cache.root))
    for start, batch, part_jobs in cache.batches(jobs, size, manifest, budget):
        params, requests = [], []
        for offset, job in enumerate(part_jobs):
            number = start + offset
            settings = {"context_width": m, "key": key, "vocab_size": vocab_size,
                        "blocked_ids": blocked, "ranges": job["ranges"],
                        "trace_path": str((batch / "trace" / f"r{number}.jsonl").resolve()),
                        "pivot_diagnostics": False}
            job["trace_path"] = settings["trace_path"]
            param = SamplingParams(n=1, temperature=temperature, top_p=1., top_k=-1, min_p=0.,
                                   max_tokens=n, ignore_eos=True, detokenize=False,
                                   seed=seed + number,
                                   logits_processors=[GumbelRequest(settings, temperature)])
            validate_params(param)
            params.append(param)
            requests.append({"prompt_token_ids": job["doc"]["prompt_ids"]})
        generated = llm.generate(requests, sampling_params=params, use_tqdm=True)
        if len(generated) != len(part_jobs):
            raise RuntimeError("vLLM output count mismatch.")
        records = []
        for job, result in zip(part_jobs, generated):
            tokens = list(result.outputs[0].token_ids)
            if list(result.prompt_token_ids) != job["doc"]["prompt_ids"] or len(tokens) != n:
                raise RuntimeError("vLLM prompt mismatch or incomplete generation.")
            audit = {row["step"]: row for row in read_jsonl(job["trace_path"])}
            records.append(source_record(job, tokens, audit, key, m))
        write_jsonl(batch / "sources.jsonl", records)
        cache.commit(batch, output / "batches" / f"b{start // size}")
        count += len(records)
        if tick_stage:
            tick(tick_stage, len(records))
        progress("本批已回传并释放本地缓存", count, len(jobs))
    # Stream the small source shards on network storage into the existing interface.
    with (output / "sources.jsonl").open("wb") as target:
        for number in range((len(jobs) + size - 1) // size):
            with (output / "batches" / f"b{number}" / "sources.jsonl").open("rb") as source:
                shutil.copyfileobj(source, target)
    cache.finish()
    return count, perf_counter() - started
