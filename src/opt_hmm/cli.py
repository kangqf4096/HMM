"""Command-line interface; base simulations do not import GPU dependencies."""
import argparse
import json
from pathlib import Path

from .io import new_output, read_json, read_jsonl
from .paths import on_colab, project_path


def main(argv=None):
    text_config = "configs/colab_arxiv.json" if on_colab() else "configs/text_smoke.json"
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
    prepare.add_argument("--config", default=text_config)
    prepare.add_argument("--model")
    prepare.add_argument("--dataset-loader", choices=["datasets", "parquet_file"],
                         help="Override the arXiv reader; parquet_file avoids datasets fragment scans")
    prepare.add_argument("--output", required=True)
    generate = sub.add_parser("generate", help="Local offline vLLM 0.8.4 V0; requires Linux/CUDA")
    generate.add_argument("--config", default=text_config)
    generate.add_argument("--model")
    generate.add_argument("--prompts", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--tensor-parallel", type=int, default=1)
    generate.add_argument("--gpu-memory", type=float, default=.8)
    generate.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    generate.add_argument('--pivot-diagnostics', action='store_true', help='Record actual pre-sampling probabilities and analytic pivot laws')
    diagnose = sub.add_parser('diagnose-pivots', help='GPU probability replay and conditional emission experiment')
    diagnose.add_argument('--input', required=True)
    diagnose.add_argument('--output', required=True)
    diagnose.add_argument('--model')
    diagnose.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    evaluate = sub.add_parser("evaluate", help="Calibrate each method and evaluate saved observations")
    evaluate.add_argument("--input", required=True, help="Generation run directory")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--config", help="Optional detector configuration; protocol settings must match")
    l0_generate = sub.add_parser("l0-generate", help="L0 sources with vLLM (Linux/CUDA): calibration/dev/test/null articles, no edits")
    l0_generate.add_argument("--config", default="configs/text_l0_smoke.json")
    l0_generate.add_argument("--model")
    l0_generate.add_argument("--prompts", required=True)
    l0_generate.add_argument("--output", required=True)
    l0_generate.add_argument("--tensor-parallel", type=int, default=1)
    l0_generate.add_argument("--gpu-memory", type=float, default=.8)
    l0_generate.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    l0_build = sub.add_parser("l0-build", help="Resample edits (HF replay of the original model) and positions.jsonl with edit-aware truth")
    l0_build.add_argument("--input", required=True, help="l0-generate run directory")
    l0_build.add_argument("--output", required=True)
    l0_build.add_argument("--model")
    l0_build.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    l0_evaluate = sub.add_parser("l0-evaluate", help="Fit, calibrate, freeze, then evaluate L0 positions (CPU only)")
    l0_evaluate.add_argument("--input", required=True, help="l0-build run directory")
    l0_evaluate.add_argument("--output", required=True)
    l0_evaluate.add_argument("--config", help="Optional JSON overriding evaluation-only settings (e.g. {\"cap\": null})")
    l0_synth = sub.add_parser("l0-synth", help="CPU pipeline test with a toy language model (not a result about OPT)")
    l0_synth.add_argument("--config", default="configs/text_l0_smoke.json")
    l0_synth.add_argument("--output", required=True)
    sub.add_parser("check-runtime", help="Check vLLM 0.8.4, CUDA, and the V0 logits processor")
    args = parser.parse_args(argv)
    # Resolve resource paths against the project even when a notebook is in /content.
    for name in ("config", "model", "input", "output", "prompts"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, project_path(value))
    if args.command == "check-runtime":
        from .runtime import adapter_selftest, check_runtime
        result = {**check_runtime(), **adapter_selftest()}
    elif args.command == 'diagnose-pivots':
        from .pivot_experiment import run as run_pivot_experiment
        result = run_pivot_experiment(args.input, args.output, args.model, args.device)
    elif args.command == "l0-generate":
        from .l0.generate import generate as l0_generate_sources
        config = read_json(args.config)
        result = l0_generate_sources(config, args.model or project_path(config["model"]), args.prompts, args.output,
                                     args.tensor_parallel, args.gpu_memory, args.dtype)
    elif args.command == "l0-build":
        from .l0.build import build as l0_build_positions
        result = l0_build_positions(args.input, args.output, model=args.model, device=args.device)
    elif args.command == "l0-evaluate":
        from .l0.evaluate import evaluate_positions
        result = evaluate_positions(args.input, args.output, read_json(args.config) if args.config else None)
    elif args.command == "l0-synth":
        from .l0.build import build as l0_build_positions
        from .l0.evaluate import evaluate_positions
        from .l0.synth import generate_sources
        base = Path(args.output)
        lm, _ = generate_sources(read_json(args.config), base / "sources")
        l0_build_positions(base / "sources", base / "positions", replay=lm.replay)
        result = evaluate_positions(base / "positions", base / "eval")
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
            if args.dataset_loader:
                config["dataset_loader"] = args.dataset_loader
            result = prepare_data(config, args.model or project_path(config["model"]), args.output)
        else:
            from .generate import generate as generate_text
            result = generate_text(config, args.model or project_path(config["model"]), args.prompts, args.output,
                                   args.tensor_parallel, args.gpu_memory, args.dtype, args.pivot_diagnostics)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
