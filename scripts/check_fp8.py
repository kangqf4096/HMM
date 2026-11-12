"""GPU smoke check of the project's V0 watermark sampler and FP8 probability replay."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VLLM_USE_V1", "0")


def generate(model_path, work, tokens):
    from vllm import LLM, SamplingParams
    from opt_hmm.io import read_json, read_jsonl, write_json
    from opt_hmm.protocol import generation_contexts
    from opt_hmm.runtime import check_runtime
    from opt_hmm.vllm_processor import GumbelRequest, validate_sampling_params

    runtime = check_runtime(model_path)
    config = read_json(model_path / "config.json")
    if (config.get("quantization_config") or {}).get("quant_method") != "fp8":
        raise ValueError("This smoke check expects the downloaded FP8 model.")
    from opt_hmm.eager_loader import EagerModelLoader
    llm = LLM(load_format=EagerModelLoader, model=str(model_path), tokenizer=str(model_path), dtype="bfloat16",
              max_model_len=1024, max_num_seqs=1, gpu_memory_utilization=.8,
              enforce_eager=True, enable_prefix_caching=False, generation_config="vllm")
    tokenizer = llm.get_tokenizer()
    vocab_size = int(llm.llm_engine.model_config.get_vocab_size())
    valid = set(tokenizer.get_vocab().values())
    if max(valid) >= vocab_size:
        raise ValueError("Tokenizer has IDs outside the model vocabulary.")
    blocked = sorted((set(range(vocab_size))-valid) | set(tokenizer.all_special_ids))
    prompt = ("The study of electromagnetic radiation in high energy collisions provides information "
              "about the properties of strongly interacting matter. In this paper we investigate "
              "the relationship between the measured photon spectrum and the underlying physical processes, "
              "using a statistical model to describe the experimental observations and their uncertainty. ")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[:50]
    params = []
    for source in ("W", "U"):
        settings = {"context_width": 4, "key": "q3-smoke-key", "vocab_size": vocab_size,
                    "blocked_ids": blocked, "ranges": [[0, tokens]] if source == "W" else [],
                    "trace_path": str(work / f"{source}.jsonl"), "pivot_diagnostics": False}
        p = SamplingParams(n=1, temperature=1., top_p=1., top_k=-1, min_p=0.,
                           max_tokens=tokens, ignore_eos=True, detokenize=False, seed=7,
                           logits_processors=[GumbelRequest(settings, 1.)])
        validate_sampling_params(p)
        params.append(p)
    outputs = llm.generate([{"prompt_token_ids": prompt_ids} for _ in params], sampling_params=params)
    samples = []
    for source, output in zip(("W", "U"), outputs):
        ids = list(output.outputs[0].token_ids)
        if len(ids) != tokens:
            raise RuntimeError("Generation did not reach the requested token length.")
        audit = {r["step"]: r for r in read_jsonl(work / f"{source}.jsonl") if r["step"] < tokens}
        if set(audit) != set(range(tokens)):
            raise RuntimeError("The V0 logits processor did not record every step.")
        contexts = generation_contexts(prompt_ids, ids, 4)
        if any(audit[t]["context"] != contexts[t] for t in range(tokens)):
            raise RuntimeError("Generation history and processor history disagree.")
        if source == "W" and any(audit[t]["chosen"] != ids[t] for t in range(tokens)):
            raise RuntimeError("Watermark winner and generated token disagree.")
        samples.append({"source": source, "token_ids": ids, "text": tokenizer.decode(ids)})
    metadata = {"dtype": "bfloat16", "vocab_size": vocab_size, "blocked_ids": blocked,
                "config": {"max_model_len": 1024, "temperature": 1.}, "runtime": runtime}
    write_json(work / "samples.json", {"prompt_ids": prompt_ids, "samples": samples})
    write_json(work / "metadata.json", metadata)
    (work / "samples.txt").write_text("\n\n".join(f"{s['source']}:\n{s['text']}" for s in samples), encoding="utf-8")
    print("Generation passed: V0 callback ran at every step; watermarked winners match output tokens.", flush=True)


def replay(model_path, work):
    import numpy as np
    from opt_hmm.io import read_json, write_json
    from opt_hmm.pivot_experiment import ModelReplay

    metadata = read_json(work / "metadata.json")
    samples = read_json(work / "samples.json")
    replay_model = ModelReplay(str(model_path), metadata)
    ids = samples["samples"][0]["token_ids"]
    p = replay_model(samples["prompt_ids"], ids)
    if p.shape != (len(ids), metadata["vocab_size"]) or not np.isfinite(p).all():
        raise RuntimeError("Replay produced an invalid probability array.")
    if not np.allclose(p.sum(axis=1), 1.) or (p < 0).any():
        raise RuntimeError("Replay probabilities are not normalized.")
    write_json(work / "passed.json", {"generation": "passed", "replay": "passed", "shape": list(p.shape)})
    print("FP8 replay passed: finite and normalized next-token probabilities.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local Qwen3 FP8 model directory")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--phase", choices=["generate", "replay"], help=argparse.SUPPRESS)
    parser.add_argument("--work", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 16 <= args.tokens <= 500:
        parser.error("--tokens must be between 16 and 500")
    model = Path(args.model).resolve()
    if args.phase:
        {"generate": generate, "replay": lambda m, w, n: replay(m, w)}[args.phase](model, args.work, args.tokens)
        return
    (ROOT / "tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="q3-", dir=ROOT / "tmp"))
    print(f"Smoke artifacts: {work}", flush=True)
    # Each phase exits before the next loads the 14B model, releasing all GPU memory.
    for phase in ("generate", "replay"):
        subprocess.run([sys.executable, __file__, str(model), "--tokens", str(args.tokens),
                        "--phase", phase, "--work", str(work)], check=True, cwd=ROOT)
    print(f"Both checks passed. Read the generated W/U text: {work / 'samples.txt'}", flush=True)
    print("This checks loading, callbacks and probability validity, not text quality or vLLM/HF numerical equality.")


if __name__ == "__main__":
    main()
