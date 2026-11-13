"""Small V0 W/U generation consistency check across batch sizes; Linux/CUDA."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VLLM_USE_V1", "0")


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def worker(args):
    from vllm import LLM, SamplingParams
    from opt_hmm.eager_loader import EagerModelLoader
    from opt_hmm.runtime import check_runtime
    from opt_hmm.vllm_processor import GumbelRequest, validate_sampling_params
    from opt_hmm.protocol import generation_contexts

    runtime = check_runtime(args.model)
    task = json.loads((args.work / "task.json").read_text())
    cfg, jobs = task["config"], task["jobs"]
    llm = LLM(model=args.model, tokenizer=args.model, load_format=EagerModelLoader,
              dtype=cfg.get("dtype", "bfloat16"), tensor_parallel_size=1,
              gpu_memory_utilization=cfg.get("gpu_memory_utilization", .8),
              max_model_len=cfg["max_model_len"], max_num_seqs=args.batch,
              max_num_batched_tokens=cfg["max_num_batched_tokens"],
              enforce_eager=cfg.get("enforce_eager", False), seed=cfg["seed"],
              enable_prefix_caching=False, generation_config="vllm")
    tokenizer = llm.get_tokenizer()
    vocab = int(llm.llm_engine.model_config.get_vocab_size())
    valid = set(tokenizer.get_vocab().values())
    blocked = sorted((set(range(vocab)) - valid) | set(tokenizer.all_special_ids))
    # Warm the decode path; exclude startup and warmup from reported throughput.
    llm.generate([{"prompt_token_ids": jobs[0]["prompt_ids"]}],
                 sampling_params=SamplingParams(max_tokens=16, ignore_eos=True,
                                                detokenize=False), use_tqdm=False)
    for repeat in range(2 if args.batch == 1 else 1):
        label = f"b{args.batch}r{repeat}"
        directory = args.work / label
        directory.mkdir()
        params = []
        for i, job in enumerate(jobs):
            settings = {"context_width": cfg["context_width"], "key": task["key"],
                        "vocab_size": vocab, "blocked_ids": blocked,
                        "ranges": [[0, args.tokens]] if job["source"] == "W" else [],
                        "trace_path": str((directory / f"{i}.jsonl").resolve()),
                        "pivot_diagnostics": False}
            p = SamplingParams(n=1, temperature=cfg.get("temperature", 1.), top_p=1.,
                               top_k=-1, min_p=0., max_tokens=args.tokens,
                               ignore_eos=True, detokenize=False, seed=job["seed"],
                               logits_processors=[GumbelRequest(settings, cfg.get("temperature", 1.))])
            validate_sampling_params(p)
            params.append(p)
        started = perf_counter()
        outputs = llm.generate([{"prompt_token_ids": j["prompt_ids"]} for j in jobs],
                               sampling_params=params, use_tqdm=True)
        seconds = perf_counter() - started
        records = []
        for i, (job, output) in enumerate(zip(jobs, outputs)):
            ids = list(output.outputs[0].token_ids)
            audit = {r["step"]: r for r in map(json.loads, (directory / f"{i}.jsonl").read_text().splitlines())}
            contexts = generation_contexts(job["prompt_ids"], ids, cfg["context_width"])
            if len(ids) != args.tokens or set(audit) != set(range(args.tokens)):
                raise RuntimeError("Incomplete tokens or callback trace")
            if any(audit[t]["context"] != contexts[t] for t in range(args.tokens)):
                raise RuntimeError("Callback history mismatch")
            if job["source"] == "W" and any(audit[t]["chosen"] != ids[t] for t in range(args.tokens)):
                raise RuntimeError("Watermark winner mismatch")
            records.append({**job, "token_ids": ids, "text": tokenizer.decode(ids)})
        save(args.work / f"{label}.json", {"seconds": seconds,
             "output_tokens_per_second": len(jobs) * args.tokens / seconds,
             "runtime": runtime, "records": records})


def compare(base, other):
    rows = []
    for a, b in zip(base["records"], other["records"]):
        x, y = a["token_ids"], b["token_ids"]
        first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), len(x))
        rows.append({"id": a["id"], "source": a["source"],
                     "identical": x == y, "common_prefix_tokens": first,
                     "first_difference_1based": first + 1 if first < len(x) else None,
                     "position_match_rate": sum(u == v for u, v in zip(x, y)) / len(x)})
    return {"output_tokens_per_second": other["output_tokens_per_second"],
            "speedup": base["seconds"] / other["seconds"], "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--prompts", type=int, default=2)
    p.add_argument("--config", type=Path, default=ROOT / "configs/text_l2_L800.json")
    p.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--batch", type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.batch:
        worker(args)
        return
    cfg = json.loads(args.config.read_text())
    if args.tokens < 16 or args.tokens + cfg["prompt_tokens"] > cfg["max_model_len"] or args.prompts < 2:
        p.error("Require >=2 prompts, >=16 output tokens, and inputs+outputs within max_model_len")
    docs = []
    roles = cfg["l2"]["sides"]["test"]["roles"]
    with (ROOT / "data/l2L800/prompts.jsonl").open(encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["role"] in roles:
                docs.append(doc)
                if len(docs) == args.prompts:
                    break
    if len(docs) != args.prompts:
        raise RuntimeError("Not enough test prompts")
    jobs = [{"id": f"{d['group']}#{s}", "source": s, "prompt_ids": d["prompt_ids"],
             "seed": cfg["seed"] + 2 * i + k}
            for i, d in enumerate(docs) for k, s in enumerate(("W", "U"))]
    (ROOT / "tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="bs-", dir=ROOT / "tmp"))
    save(work / "task.json", {"config": cfg, "jobs": jobs, "tokens": args.tokens,
                              "key": cfg["l2"]["sides"]["test"]["key"]})
    print(f"Artifacts: {work}", flush=True)
    for batch in (1, 2, 4):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), args.model,
                        "--work", str(work), "--batch", str(batch),
                        "--tokens", str(args.tokens)], check=True, cwd=ROOT)
    base = json.loads((work / "b1r0.json").read_text(encoding="utf-8"))
    report = {name: compare(base, json.loads((work / f"{name}.json").read_text(encoding="utf-8")))
              for name in ("b1r1", "b2r0", "b4r0")}
    save(work / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {work / 'report.json'}")
    print("Small-sample token consistency only; not an accuracy or detector-metric estimate.")


if __name__ == "__main__":
    main()
