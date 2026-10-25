"""Offline local vLLM generation, audited seeds, and edit-aware signal truth."""
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import validate_text
from .data import DATASET, REVISION, SUBSET
from .edits import edit
from .io import new_output, provenance, read_json, read_jsonl, write_json, write_jsonl
from .protocol import (PROTOCOL, PUBLIC_KEY, context_seed, extract_pivots,
                       generation_contexts, operational_labels, pivot)
from .runtime import check_runtime


def _record(doc, original, tokens, origin, variant, config, key, actions=None):
    started = perf_counter()
    y, mask, seeds = extract_pivots(tokens, config["context_width"], key)
    source, z = operational_labels(tokens, origin, original["token_ids"], original["contexts"],
                                   original["enabled"], config["context_width"])
    available_seeds = [s for s in seeds if s is not None]
    repeats = len(available_seeds) - len(set(available_seeds))
    return {"id": f"{doc['group']}/{variant}", "group": doc["group"], "role": doc["role"],
            "variant": variant, "token_ids": tokens, "origin": origin, "o": source, "z": z,
            "mask": mask, "y": [float(v) if valid else None for v, valid in zip(y, mask)],
            "verification_seeds": seeds, "edit_actions": actions or [],
            "duplicate_context_count": repeats, "pivot_seconds": perf_counter() - started}


def generate(config, model, prompts_dir, output, tensor_parallel=1, gpu_memory=.8, dtype="float16",
             pivot_diagnostics=False):
    validate_text(config)
    if tensor_parallel < 1 or not 0 < gpu_memory <= 1:
        raise ValueError('Invalid tensor parallelism or GPU memory fraction.')
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
        raise ValueError("Tokenized prompts belong to a different model. Prepare again with the new model.")
    runtime = check_runtime()
    output = new_output(output)
    trace_dir = output / "trace"
    trace_dir.mkdir()
    prompts = read_jsonl(Path(prompts_dir) / "prompts.jsonl")
    pool_docs = [d for d in prompts if d["role"] == "pool"]
    pool = [t for d in pool_docs for t in d["pool_ids"]]
    from vllm import LLM, SamplingParams
    key = os.environ.get("OPT_HMM_KEY", PUBLIC_KEY)
    n, m = config["tokens"], config["context_width"]
    if n <= m or config.get("temperature", 1.) <= 0:
        raise ValueError("Need tokens > context width and positive temperature.")
    variants = config.get("edits", [{"kind": "replace", "rate": .1}])
    lengths = {n}
    for attack in variants:
        count = int(attack["rate"] * n)
        lengths.add(n + count if attack["kind"] == "insert" else
                    n - count if attack["kind"] == "delete" else n)
    if min(lengths) <= m:
        raise ValueError("Edited sequence is too short to evaluate.")
    max_model_len = config.get("max_model_len", 1024)
    if config["prompt_tokens"] + max(lengths) > max_model_len:
        raise ValueError("Prompt plus generation exceeds max_model_len.")
    started = perf_counter()
    llm = LLM(model=model, tokenizer=model, dtype=dtype, tensor_parallel_size=tensor_parallel,
              gpu_memory_utilization=gpu_memory, max_model_len=max_model_len,
              max_num_seqs=config.get("max_num_seqs", 8), seed=config.get("seed", 20261015),
              enforce_eager=True, enable_prefix_caching=False, generation_config="vllm",
              async_scheduling=False,
              logits_processors=["opt_hmm.vllm_processor:GumbelMaxProcessor"])
    model_load_seconds = perf_counter() - started
    tokenizer = llm.get_tokenizer()
    vocab_size = int(llm.model_config.get_vocab_size())
    valid_ids = set(tokenizer.get_vocab().values())
    if max(valid_ids) >= vocab_size:
        raise ValueError("Tokenizer has added IDs outside model output vocabulary; reconcile model/tokenizer.")
    blocked = sorted((set(range(vocab_size)) - valid_ids) | set(tokenizer.all_special_ids))
    if not valid_ids - set(blocked):
        raise ValueError("No ordinary tokens remain.")
    pool = [t for t in pool if t not in blocked]
    if not pool:
        raise ValueError("Empty ordinary token pool.")
    jobs, params, requests = [], [], []
    for doc in prompts:
        if doc["role"] == "pool":
            continue
        modes = ["null"] if doc["role"] in ("null_cal", "null_test") else ["watermarked"]
        if doc["role"] in ("dev", "test"):
            modes.append("switch")
        for mode in modes:
            requested_length = max(lengths) if mode == "null" else n
            if mode == "watermarked":
                ranges = [[0, requested_length]]
            elif mode == "switch":
                ranges = [[int(a * n), int(b * n)] for a, b in config.get("switch_ranges", [[.2, .7]])]
            else:
                ranges = []
            trace_path = str((trace_dir / f"r{len(jobs)}.jsonl").resolve())
            settings = {"context_width": m, "key": key, "vocab_size": vocab_size,
                        "blocked_ids": blocked, "ranges": ranges, "trace_path": trace_path,
                        "pivot_diagnostics": pivot_diagnostics}
            params.append(SamplingParams(n=1, temperature=config.get("temperature", 1.),
                                         top_p=1., top_k=-1, min_p=0.,
                                         max_tokens=requested_length, ignore_eos=True,
                                         detokenize=False, seed=config.get("seed", 20261015) + len(jobs),
                                         extra_args={"opt_hmm": settings}))
            requests.append({"prompt_token_ids": doc["prompt_ids"]})
            jobs.append((doc, mode, requested_length, ranges, trace_path))
    started = perf_counter()
    outputs = llm.generate(requests, sampling_params=params, use_tqdm=True)
    generation_seconds = perf_counter() - started
    if len(outputs) != len(jobs):
        raise RuntimeError("vLLM output count mismatch.")
    records, sources = [], []
    edit_rng = np.random.default_rng(config.get("seed", 20261015) + 700001)
    for job, generated in zip(jobs, outputs):
        doc, mode, requested_length, ranges, trace_path = job
        if list(generated.prompt_token_ids) != doc["prompt_ids"]:
            raise RuntimeError("vLLM request ordering/prompt mismatch.")
        tokens = list(generated.outputs[0].token_ids)
        if len(tokens) != requested_length:
            raise RuntimeError(f"Incomplete generation for {doc['group']}: {len(tokens)} != {requested_length}")
        contexts = generation_contexts(doc["prompt_ids"], tokens, m)
        enabled = [any(a <= t < b for a, b in ranges) for t in range(len(tokens))]
        seeds = [str(context_seed(c, key)) for c in contexts]
        audit = {}
        for row in read_jsonl(trace_path):
            audit[row["step"]] = row
        if set(audit) != set(range(len(tokens))):
            raise RuntimeError("Missing generation audit steps; logits processor did not run as expected.")
        for t, row in audit.items():
            if (row["seed"] != seeds[t] or row["context"] != contexts[t] or row["active"] != enabled[t]):
                raise RuntimeError("Generation/verification context or seed mismatch.")
            if enabled[t] and (row["chosen"] != tokens[t] or
                               row["selected_y"] != pivot(int(seeds[t]), tokens[t])):
                raise RuntimeError("Watermark sampler audit does not match emitted tokens.")
        original = {"id": f"{doc['group']}/{mode}", "group": doc["group"], "role": doc["role"],
                    "token_ids": tokens, "prompt_ids": doc["prompt_ids"],
                    "enabled": enabled, "contexts": contexts, "seeds": seeds,
                    "probability_trace": [audit[t].get('probability') for t in range(len(tokens))]
                    if pivot_diagnostics else None}
        sources.append(original)
        if mode == "null":
            for length in sorted(lengths):
                records.append(_record(doc, original, tokens[:length], list(range(length)),
                                       f"null_n{length}", config, key))
        else:
            records.append(_record(doc, original, tokens, list(range(len(tokens))), mode, config, key))
            if mode == "watermarked" and doc["role"] in ("test", "dev"):
                for attack in variants:
                    final, origin, actions = edit(tokens, attack["kind"], attack["rate"], pool, edit_rng)
                    variant = f"{attack['kind']}_{attack['rate']:g}"
                    records.append(_record(doc, original, final, origin, variant, config, key, actions))
    write_jsonl(output / "sources.jsonl", sources)
    write_jsonl(output / "observations.jsonl", records)
    write_json(output / "metadata.json", {"config": config, "model": model,
                                        "runtime": runtime, "provenance": provenance(),
                                        "protocol": PROTOCOL, "key_id": config.get("key_id", "public-research-v1"),
                                        "public_research_key": key == PUBLIC_KEY,
                                        "vocab_size": vocab_size, "blocked_ids": blocked,
                                        "tensor_parallel": tensor_parallel, "dtype": dtype,
                                        "gpu_memory_utilization": gpu_memory,
                                        "model_load_seconds": model_load_seconds,
                                        "generation_seconds": generation_seconds,
                                        "pivot_diagnostics": pivot_diagnostics,
                                        "audit": "Every active chosen token, pivot, context and seed verified.",
                                        "lengths": sorted(lengths),
                                        "prompt_manifest": manifest})
    return {"requests": len(jobs), "records": len(records), "generation_seconds": generation_seconds}
