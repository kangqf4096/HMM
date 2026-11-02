"""Offline vLLM 0.8.4 V0 generation of L0 sources (Linux/CUDA only; no edits, no replay)."""
import os
from pathlib import Path
from time import perf_counter

from ..data import DATASET, REVISION, SUBSET
from ..io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from ..paths import project_path
from ..progress import progress
from ..protocol import PROTOCOL, PUBLIC_KEY, context_seed, generation_contexts, pivot
from .config import validate_l0
from .schema import scenario_for

os.environ.setdefault("VLLM_USE_V1", "0")


def plan_jobs(prompts, config):
    """One generation job per non-pool article; scenarios are fixed by role order."""
    n, seen, jobs = config["tokens"], {}, []
    for doc in prompts:
        role = doc["role"]
        if role == "pool":
            continue
        index = seen.get(role, 0)
        seen[role] = index + 1
        scenario = scenario_for(role, index)
        if scenario == "switch":
            ranges = [[int(a * n), int(b * n)] for a, b in config["switch_ranges"]]
        else:
            ranges = [] if scenario == "null" else [[0, n]]
        jobs.append({"doc": doc, "scenario": scenario, "ranges": ranges})
    return jobs


def source_record(job, tokens, audit, key, context_width):
    """Verify the sampler audit against the emitted tokens and build the source row."""
    doc, ranges = job["doc"], job["ranges"]
    contexts = generation_contexts(doc["prompt_ids"], tokens, context_width)
    seeds = [str(context_seed(c, key)) for c in contexts]
    enabled = [any(a <= t < b for a, b in ranges) for t in range(len(tokens))]
    if set(audit) != set(range(len(tokens))):
        raise RuntimeError("Missing generation audit steps; logits processor did not run as expected.")
    for t, row in audit.items():
        if row["seed"] != seeds[t] or row["context"] != contexts[t] or row["active"] != enabled[t]:
            raise RuntimeError("Generation/verification context or seed mismatch.")
        if enabled[t] and (row["chosen"] != tokens[t] or row["selected_y"] != pivot(int(seeds[t]), tokens[t])):
            raise RuntimeError("Watermark sampler audit does not match emitted tokens.")
    return {"article_id": doc["group"], "split": doc["role"], "scenario": job["scenario"],
            "ranges": ranges, "prompt_ids": doc["prompt_ids"], "token_ids": [int(t) for t in tokens],
            "source_split": doc.get("source_split"), "source_row": doc.get("source_row")}


def generate(config, model, prompts_dir, output, tensor_parallel=1, gpu_memory=.8, dtype="float16"):
    model, prompts_dir, output = (project_path(p) for p in (model, prompts_dir, output))
    model = str(model)
    validate_l0(config)
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
    runtime = check_runtime()
    output = new_output(output)
    trace_dir = output / "trace"
    trace_dir.mkdir()
    jobs = plan_jobs(read_jsonl(Path(prompts_dir) / "prompts.jsonl"), config)
    progress("加载 vLLM")
    from vllm import LLM, SamplingParams
    key = os.environ.get("OPT_HMM_KEY", PUBLIC_KEY)
    n, m, temperature = config["tokens"], config["context_width"], config.get("temperature", 1.)
    seed = config.get("seed", 20261015)
    started = perf_counter()
    llm = LLM(model=model, tokenizer=model, dtype=dtype, tensor_parallel_size=tensor_parallel,
              gpu_memory_utilization=gpu_memory, max_model_len=config["max_model_len"],
              max_num_seqs=config["max_num_seqs"], seed=seed, enforce_eager=config.get("enforce_eager", False),
              enable_prefix_caching=False, generation_config="vllm")
    model_load_seconds = perf_counter() - started
    tokenizer = llm.get_tokenizer()
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    valid_ids = set(tokenizer.get_vocab().values())
    if max(valid_ids) >= vocab_size:
        raise ValueError("Tokenizer has added IDs outside model output vocabulary; reconcile model/tokenizer.")
    blocked = sorted((set(range(vocab_size)) - valid_ids) | set(tokenizer.all_special_ids))
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
    outputs = llm.generate(requests, sampling_params=params, use_tqdm=True)
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
    write_json(output / "metadata.json", {
        "config": config, "model": model, "runtime": runtime, "provenance": provenance(),
        "protocol": PROTOCOL, "key_id": config.get("key_id", "public-research-v1"),
        # Research phase: the plaintext key keeps a run reproducible; see docs/reproducibility_and_secrets_20261101.md.
        "key": key, "public_research_key": key == PUBLIC_KEY, "vocab_size": vocab_size,
        "blocked_ids": blocked, "tensor_parallel": tensor_parallel, "dtype": dtype,
        "gpu_memory_utilization": gpu_memory, "enforce_eager": config.get("enforce_eager", False),
        "model_load_seconds": model_load_seconds, "generation_seconds": generation_seconds,
        "audit": "Every active chosen token, pivot, context and seed verified.", "prompt_manifest": manifest})
    return {"articles": len(sources), "generation_seconds": generation_seconds}
