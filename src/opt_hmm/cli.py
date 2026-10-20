"""Command-line interface; base simulations do not import GPU dependencies."""
import argparse
import json
from pathlib import Path

from .io import new_output, read_json, read_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(prog="opt-hmm")
    sub = parser.add_subparsers(dest="command", required=True)
    simulate = sub.add_parser("simulate", help="CPU synthetic experiments with independent null calibration")
    simulate.add_argument("--config", default="configs/sim_smoke.json")
    simulate.add_argument("--output", required=True)
    simulate.add_argument("--scenario", choices=["markov", "iid", "island", "hole", "strong_weak"])
    simulate.add_argument("--tokens", type=int)
    simulate.add_argument("--test-documents", type=int)
    simulate.add_argument("--signal-p", type=float, nargs="+")
    simulate.add_argument("--segment-length", type=int)
    prepare = sub.add_parser("prepare", help="Download scientific arXiv papers and freeze document-level prompt roles")
    prepare.add_argument("--config", default="configs/text_smoke.json")
    prepare.add_argument("--model")
    prepare.add_argument("--output", required=True)
    generate = sub.add_parser("generate", help="Local offline vLLM 0.19.1; requires Linux/CUDA")
    generate.add_argument("--config", default="configs/text_smoke.json")
    generate.add_argument("--model")
    generate.add_argument("--prompts", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--tensor-parallel", type=int, default=1)
    generate.add_argument("--gpu-memory", type=float, default=.8)
    generate.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    evaluate = sub.add_parser("evaluate", help="Calibrate each method and evaluate saved observations")
    evaluate.add_argument("--input", required=True, help="Generation run directory")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config", help="Optional detector configuration; protocol settings must match")
    sub.add_parser("check-runtime", help="Check actual vLLM version, CUDA, and batch adapter")
    args = parser.parse_args(argv)
    if args.command == "check-runtime":
        from .runtime import adapter_selftest, check_runtime
        result = {**check_runtime(), **adapter_selftest()}
    elif args.command == "evaluate":
        from .experiment import run_evaluation
        metadata = read_json(Path(args.input) / "metadata.json")
        config = dict(metadata["config"])
        if args.config:
            override = read_json(args.config)
            for name in ("tokens", "context_width", "prompt_tokens", "temperature", "key_id"):
                if override.get(name, config.get(name)) != config.get(name):
                    raise ValueError(f"Saved generation and evaluation differ on {name}.")
            config.update(override)
        result = run_evaluation(read_jsonl(Path(args.input) / "observations.jsonl"),
                                new_output(args.output), config)
        result = {"report": str(Path(args.output) / "report.md"),
                  "paired_iou_difference_vs_independent": result["paired_iou_difference_vs_independent"]}
    else:
        config = read_json(args.config)
        if args.command == "simulate":
            from .simulate import run
            for name in ("scenario", "tokens", "test_documents", "signal_p", "segment_length"):
                if getattr(args, name, None) is not None:
                    config[name] = getattr(args, name)
            result = run(config, args.output)
            result = {"report": str(Path(args.output) / "report.md"),
                      "paired_iou_difference_vs_independent": result["paired_iou_difference_vs_independent"]}
        elif args.command == "prepare":
            from .data import prepare as prepare_data
            result = prepare_data(config, args.model or config["model"], args.output)
        else:
            from .generate import generate as generate_text
            result = generate_text(config, args.model or config["model"], args.prompts, args.output,
                                   args.tensor_parallel, args.gpu_memory, args.dtype)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
