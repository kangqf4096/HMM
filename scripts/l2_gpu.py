"""Resume the L2 GPU pipeline between complete stages; restart incomplete stages."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from l2_stage import archive_incomplete, read_json, restore_progress, same_config

ROOT = Path(os.environ.get("OPT_HMM_PROJECT") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))


def same_model(saved, model):
    source_file = model / "opt_hmm_source.json"
    recorded = saved.get("model_source") or (saved.get("prompt_manifest") or {}).get("model_source")
    if recorded and source_file.is_file():
        current = read_json(source_file)
        return all(recorded.get(k) and recorded[k] == current.get(k) for k in ("repo_id", "revision"))
    return Path(saved.get("model", "")).resolve() == model


def complete(output, files, config, model, side=None, dtype=None):
    if not all((output / name).is_file() and (output / name).stat().st_size for name in files):
        return False
    try:
        saved = read_json(output / files[-1])  # The final manifest is written after the data.
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(saved, dict) or not isinstance(saved.get("config"), dict):
        return False
    if side is not None and saved.get("side") is None:
        return False
    mismatch = not same_config(saved["config"], config)
    if side is not None:
        mismatch |= saved.get("side") != side
    if "model" in saved:
        mismatch |= not same_model(saved, model)
    if dtype is not None:
        mismatch |= saved.get("dtype") != dtype or saved.get("tensor_parallel", 1) != 1
    if mismatch:
        raise ValueError(f"Saved stage differs in config, model, side or dtype: {output}. Use a new run-name.")
    return True


def run_stage(name, output, files, args, config, model, progress, *, side=None,
              dtype=None, stage=None, force=False):
    finished = complete(output, files, config, model, side, dtype)
    if finished and not force:
        print(f"[skip] {name}: {output}", flush=True)
        if stage:
            restore_progress(config, progress, stage)
        return False
    archive_incomplete(output)
    print(f"[run] {name}: {output}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), *map(str, args)], check=True, cwd=ROOT)
    if not complete(output, files, config, model, side, dtype):
        raise RuntimeError(f"Stage returned without complete outputs: {output}")
    return True


def run_pipeline(model, run, config_path, progress, dtype=None):
    if not run or run in (".", "..") or "/" in run or "\\" in run:
        raise ValueError("run-name must be a single directory name.")
    model = Path(model).resolve()
    config = read_json(config_path)
    dtype = dtype or config.get("dtype", "float16")
    progress = Path(progress).resolve()
    prompts = ROOT / "data" / run
    prepared = run_stage("prompts", prompts, ("prompts.jsonl", "manifest.json"),
                         ["prepare", "--config", config_path, "--model", model, "--output", prompts],
                         config, model, progress)
    train_rebuilt = False
    for side in ("train", "test"):
        generated = ROOT / "runs" / f"{run}-{side}-gen"
        built = ROOT / "runs" / f"{run}-{side}-pos"
        changed = run_stage(f"{side} generation", generated, ("sources.jsonl", "metadata.json"),
                            ["l2-generate", "--config", config_path, "--model", model,
                             "--prompts", prompts, "--side", side, "--output", generated,
                             "--dtype", dtype, "--progress", progress], config, model, progress,
                            side=side, dtype=dtype, stage="G1" if side == "train" else "G2", force=prepared)
        build_args = ["l2-build", "--input", generated, "--output", built, "--model", model,
                      "--progress", progress]
        if side == "test":
            build_args += ["--train", ROOT / "runs" / f"{run}-train-pos"]
        rebuilt = run_stage(f"{side} build", built, ("positions.jsonl", "articles.jsonl", "manifest.json"),
                            build_args, config, model, progress, side=side,
                            stage="B1" if side == "train" else "B2",
                            force=changed or (side == "test" and train_rebuilt))
        if side == "train":
            train_rebuilt = rebuilt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    run_pipeline(args.model, args.run, args.config, args.progress, args.dtype)


if __name__ == "__main__":
    main()
