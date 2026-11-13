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
    l0_evaluate.add_argument("--stage", choices=["s1", "s2", "s3", "s3b", "s4"],
                             help="L1 stage (plan_general_transition); without it the old L0 evaluation runs")
    l0_evaluate.add_argument("--reuse-seedbs", help="Earlier L1 run directory whose seedbs_params.json is reused")
    l0_evaluate.add_argument("--methods", nargs="+", help="Restrict an L1 stage to these methods")
    l0_evaluate.add_argument("--param-star", help="Parametric family chosen at s3, for the s3b nocap run")
    l0_evaluate.add_argument("--workers", type=int, default=1, help="Processes for the seedbs bootstrap")
    l0_synth = sub.add_parser("l0-synth", help="CPU pipeline test with a toy language model (not a result about OPT)")
    l0_synth.add_argument("--config", default="configs/text_l0_smoke.json")
    l0_synth.add_argument("--output", required=True)
    l2_generate = sub.add_parser("l2-generate", help="L2 sources with vLLM (Linux/CUDA) for one side: W and U generations")
    l2_generate.add_argument("--config", default="configs/text_l2.json")
    l2_generate.add_argument("--model")
    l2_generate.add_argument("--prompts", required=True)
    l2_generate.add_argument("--output", required=True)
    l2_generate.add_argument("--side", required=True, choices=["train", "test"])
    l2_generate.add_argument("--tensor-parallel", type=int, default=1)
    l2_generate.add_argument("--gpu-memory", type=float,
                             help="Override config.gpu_memory_utilization (default: 0.8 when not configured)")
    l2_generate.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                             help="Override config.dtype (default: float16 when not configured)")
    l2_generate.add_argument("--progress")
    l2_build = sub.add_parser("l2-build", help="L2 positions: compose W/U, resample replay, pivots and truth")
    l2_build.add_argument("--input", required=True, help="l2-generate run directory")
    l2_build.add_argument("--output", required=True)
    l2_build.add_argument("--model")
    l2_build.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    l2_build.add_argument("--train", help="Training-side build directory (test side: reads L_top from it)")
    l2_build.add_argument("--progress")
    l2_mechanism = sub.add_parser("l2-mechanism", help="Mechanism check M1/M2 on the training side (CPU)")
    l2_mechanism.add_argument("--input", required=True, help="Training-side l2-build directory")
    l2_mechanism.add_argument("--output", required=True)
    l2_mechanism.add_argument("--progress")
    l2_fit = sub.add_parser("l2-fit", help="L2 estimation, dev selection and freeze on the training side (CPU)")
    l2_fit.add_argument("--input", required=True, help="Training-side l2-build directory")
    l2_fit.add_argument("--output", required=True)
    l2_fit.add_argument("--workers", type=int, default=1)
    l2_fit.add_argument("--cache", help="seedbs precompute cache directory")
    l2_fit.add_argument("--progress")
    l2_test = sub.add_parser("l2-test", help="One L2 test run from frozen.json on the test side (CPU)")
    l2_test.add_argument("--frozen", required=True, help="l2-fit output directory (or frozen.json)")
    l2_test.add_argument("--input", required=True, help="Test-side l2-build directory")
    l2_test.add_argument("--output", required=True)
    l2_test.add_argument("--workers", type=int, default=1)
    l2_test.add_argument("--cache")
    l2_test.add_argument("--mechanism", help="mechanism.json of the training side")
    l2_test.add_argument("--progress")
    l2_extra = sub.add_parser("l2-extra", help="Beta-shaped f1(e') (fb) and the SPOT baseline on a finished L2 run (CPU)")
    l2_extra.add_argument("--frozen", required=True, help="Reference l2-fit directory (or frozen.json)")
    l2_extra.add_argument("--train", required=True, help="Training-side l2-build directory")
    l2_extra.add_argument("--input", required=True, help="Test-side l2-build directory")
    l2_extra.add_argument("--reference", required=True, help="Reference l2-test directory (metrics.jsonl)")
    l2_extra.add_argument("--output", required=True)
    l2_cap = sub.add_parser("l2-cap", help="m100 cap sensitivity after the test run")
    l2_cap.add_argument("--frozen", required=True)
    l2_cap.add_argument("--train", required=True)
    l2_cap.add_argument("--input", required=True, help="Test-side l2-build directory")
    l2_cap.add_argument("--output", required=True)
    l2_cap.add_argument("--progress")
    l2_derive = sub.add_parser("l2-derive", help="Length study: derive prompts or sources at another text length, "
                                                 "or check that a longer run's prefix equals a shorter run")
    l2_derive.add_argument("kind", choices=["prompts", "sources", "prefix"])
    l2_derive.add_argument("--input", required=True, help="Source run directory (the longer run for 'prefix')")
    l2_derive.add_argument("--output", help="Derived directory (prompts, sources) or JSON report (prefix)")
    l2_derive.add_argument("--config", help="Configuration of the derived run (prompts, sources)")
    l2_derive.add_argument("--short", help="Shorter run's generation directory ('prefix')")
    l2_degenerate = sub.add_parser("l2-degenerate", help="Length study: degenerate prompt groups in built *-pos directories")
    l2_degenerate.add_argument("--input", nargs="+", required=True, help="Built *-pos directories (any side, any length)")
    l2_degenerate.add_argument("--output", required=True, help="JSON file with the union of degenerate groups")
    l2_filter = sub.add_parser("l2-filter", help="Length study: drop prompt groups from a built side")
    l2_filter.add_argument("--input", required=True, help="Built *-pos directory")
    l2_filter.add_argument("--output", required=True)
    l2_filter.add_argument("--groups", required=True, help="JSON file written by l2-degenerate")
    l2_filter.add_argument("--config", help="Configuration written into the manifest (same for both sides)")
    l2_reuse = sub.add_parser("l2-reuse", help="Length study, group B: frozen f1 from another length, refit a, pi, tau (CPU)")
    l2_reuse.add_argument("--frozen", required=True, help="Fit directory (or frozen.json) whose f1 is reused")
    l2_reuse.add_argument("--train", required=True, help="Training-side *-pos directory of the new length")
    l2_reuse.add_argument("--base", required=True, help="Group A fit directory of the new length (SeedBS, keys, L_top)")
    l2_reuse.add_argument("--output", required=True)
    l2_lengths = sub.add_parser("l2-lengths", help="Length study report: group A (retrained) against group B (frozen f1)")
    l2_lengths.add_argument("--runs", nargs="+", required=True, help="Run names, e.g. l2L300f l2q1f l2L800f")
    l2_lengths.add_argument("--reference-tokens", type=int, default=500)
    l2_lengths.add_argument("--degenerate", help="JSON file written by l2-degenerate")
    l2_lengths.add_argument("--window-edges", type=int, nargs="+", default=[150, 300, 500],
                            help="Token positions that cut the longest text into windows for the f1 position analysis")
    l2_lengths.add_argument("--output", required=True)
    l2_synth = sub.add_parser("l2-synth", help="L2 pipeline test with a toy language model (not a result about OPT)")
    l2_synth.add_argument("--config", default="configs/text_l2_smoke.json")
    l2_synth.add_argument("--output", required=True)
    l2_synth.add_argument("--workers", type=int, default=1)
    l2_weights = sub.add_parser("l2-weights", help="Seconds per unit of each stage from a finished progress file")
    l2_weights.add_argument("--progress", required=True)
    l2_weights.add_argument("--output", required=True)
    l2_stages = sub.add_parser("l2-stages", help="Print the overall progress stage table of an L2 config")
    l2_stages.add_argument("--config", default="configs/text_l2.json")
    runtime_check = sub.add_parser("check-runtime", help="Check vLLM 0.8.4/0.8.5, CUDA, and the V0 logits processor")
    runtime_check.add_argument("--model", help="Also check the local model's vLLM version requirements")
    args = parser.parse_args(argv)
    # Resolve resource paths against the project even when a notebook is in /content.
    for name in ("config", "model", "input", "output", "prompts", "frozen", "train", "cache", "mechanism", "progress",
                 "short", "base", "groups", "degenerate"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, [project_path(v) for v in value] if isinstance(value, list) else project_path(value))
    if args.command == "check-runtime":
        from .runtime import adapter_selftest, check_runtime
        result = {**check_runtime(args.model), **adapter_selftest()}
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
        override = read_json(args.config) if args.config else None
        if args.stage:
            from .l0.evaluate_v2 import run_stage
            result = run_stage(args.input, args.output, args.stage, override, args.reuse_seedbs, args.workers,
                               args.methods, args.param_star)
        else:
            from .l0.evaluate import evaluate_positions
            result = evaluate_positions(args.input, args.output, override)
    elif args.command == "l0-synth":
        from .l0.build import build as l0_build_positions
        from .l0.evaluate import evaluate_positions
        from .l0.synth import generate_sources
        base = Path(args.output)
        lm, _ = generate_sources(read_json(args.config), base / "sources")
        l0_build_positions(base / "sources", base / "positions", replay=lm.replay)
        result = evaluate_positions(base / "positions", base / "eval")
    elif args.command == "l2-generate":
        from .l2.generate import generate as l2_generate_sources
        config = read_json(args.config)
        result = l2_generate_sources(config, args.model or project_path(config["model"]), args.prompts, args.output,
                                     args.side, args.tensor_parallel, args.gpu_memory, args.dtype, args.progress)
    elif args.command == "l2-build":
        from .l2.build import build as l2_build_positions
        result = l2_build_positions(args.input, args.output, model=args.model, device=args.device, train_dir=args.train,
                                     progress_path=args.progress)
    elif args.command == "l2-mechanism":
        from .l2.mechanism import run_mechanism
        result = run_mechanism(args.input, args.output, args.progress)
    elif args.command == "l2-fit":
        from .l2.fit import run_fit
        result = run_fit(args.input, args.output, args.workers, args.cache, args.progress)
    elif args.command == "l2-test":
        from .l2.evaluate import run_test
        result = run_test(args.frozen, args.input, args.output, args.workers, args.cache, args.progress,
                          mechanism=args.mechanism)
    elif args.command == "l2-extra":
        from .l2.extra import run_extra
        result = run_extra(args.frozen, args.train, args.input, args.reference, args.output)
    elif args.command == "l2-cap":
        from .l2.capsweep import run_cap_sweep
        result = run_cap_sweep(args.frozen, args.train, args.input, args.output, args.progress)
    elif args.command == "l2-derive":
        from .l2 import derive
        if args.kind == "prefix":
            if not args.short:
                raise ValueError("prefix needs --short.")
            result = derive.check_prefix(args.input, args.short, args.output)
        else:
            if not args.config or not args.output:
                raise ValueError(f"{args.kind} needs --config and --output.")
            run = derive.derive_prompts if args.kind == "prompts" else derive.derive_sources
            result = run(args.input, args.output, read_json(args.config))
    elif args.command == "l2-degenerate":
        from .l2.degenerate import run_degenerate
        result = run_degenerate(args.input, args.output)
    elif args.command == "l2-filter":
        from .l2.filter import filter_side
        config = read_json(args.config) if args.config else None
        result = filter_side(args.input, args.output, read_json(args.groups)["groups"], config)
    elif args.command == "l2-reuse":
        from .l2.reuse import run_reuse
        result = run_reuse(args.frozen, args.train, args.base, args.output)
    elif args.command == "l2-lengths":
        from .l2.lengths import run_lengths
        result = run_lengths(args.runs, args.output, args.reference_tokens, args.degenerate, args.window_edges)
    elif args.command == "l2-synth":
        from .l2.synth import run_synth
        result = run_synth(read_json(args.config), args.output, workers=args.workers)
    elif args.command == "l2-weights":
        from .l2.stages import weights_from_progress
        result = weights_from_progress(args.progress, args.output)
    elif args.command == "l2-stages":
        from .l2.stages import compare_to_plan, stage_table
        config = read_json(args.config)
        compare_to_plan(config)
        result = {k: v["total"] for k, v in stage_table(config).items()}
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
