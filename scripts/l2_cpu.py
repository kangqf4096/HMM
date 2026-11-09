"""Resume CPU stages using complete results and a separate persistent SeedBS cache."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from l2_stage import archive_incomplete, read_json, restore_progress, same_config, write_json

ROOT = Path(os.environ.get("OPT_HMM_PROJECT") or Path(__file__).resolve().parents[1]).resolve()
sys.path.insert(0, str(ROOT / "src"))

FILES = {
    "mechanism": ("mechanism.json", "mechanism.md"),
    "fit": ("coverage.json", "frozen.json", "fit_report.json", "fit_report.md", "provenance.json"),
    "test": ("report.json", "report.md", "metrics.jsonl"),
    "cap": ("cap_sweep.json", "cap_sweep.md"),
}
STAGES = {"mechanism": ("D1",), "fit": ("F1", "E1", "S1", "S2", "T1", "BS"),
          "test": ("R1", "REP"), "cap": ("C1",)}


def inputs_for(run):
    manifests = {}
    for side in ("train", "test"):
        directory = ROOT / "runs" / f"{run}-{side}-pos"
        for name in ("manifest.json", "positions.jsonl", "articles.jsonl"):
            if not (directory / name).is_file() or not (directory / name).stat().st_size:
                raise FileNotFoundError(f"Missing or empty GPU input: {directory / name}. Keep both complete *-pos directories.")
        manifest = read_json(directory / "manifest.json")
        if manifest.get("side") != side:
            raise ValueError(f"Wrong input side: {directory}")
        manifests[side] = manifest
    train, test = manifests["train"], manifests["test"]
    if not same_config(train["config"], test["config"]) or train.get("l_top") != test.get("l_top"):
        raise ValueError("Training and test builds have different configurations or L_top.")
    return manifests


def check_frozen(frozen, train):
    config = train["config"]
    if (frozen.get("train_key_id") != train["key_id"] or frozen.get("l_top") != train.get("l_top") or
            any(config.get(k) != v for k, v in frozen.get("config", {}).items()) or
            any(config["l2"].get(k) != v for k, v in frozen.get("l2", {}).items())):
        raise ValueError("Saved frozen parameters belong to different training data/configuration. Use a new run-name.")


def complete(name, output, context, train):
    if not all((output / file).is_file() and (output / file).stat().st_size for file in FILES[name]):
        return False
    try:
        documents = {file: read_json(output / file) for file in FILES[name] if file.endswith(".json")}
        record = read_json(output / "resume.json") if (output / "resume.json").is_file() else None
        if record is not None:
            if record.get("inputs") != context["inputs"]:
                raise ValueError(f"CPU inputs changed for {output}; use a new run-name.")
            if record != context:
                return False
        if name == "fit":
            frozen = documents["frozen.json"]
            if not frozen.get("config") or not frozen.get("l2") or not frozen.get("methods") or not frozen.get("seedbs"):
                return False
            check_frozen(frozen, train)
        elif name == "test":
            report = documents["report.json"]
            # metrics.jsonl is the last legacy output; reject an interrupted final write.
            expected = context["inputs"]["test"]["articles"] * len(report["methods"])
            count, last = 0, ""
            with (output / "metrics.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        count, last = count + 1, line
            if count != expected or not last or not isinstance(json.loads(last), dict):
                return False
        elif name == "mechanism":
            if not {"groups", "M1", "M2"} <= documents["mechanism.json"].keys():
                return False
        elif name == "cap":
            report = documents["cap_sweep.json"]
            if len(report["rows"]) != 2 * len(train["config"]["l2"]["cap_sweep"]):
                return False
            frozen = context["dependencies"]["frozen"]
            if report["frozen_cap"] != {f"{mode}_m100": frozen["methods"][f"{mode}_m100"]["cap"] for mode in ("pd", "pe")}:
                return False
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        return False
    return True


def import_legacy_cache(output, cache):
    # Old versions put reusable cache inside the output directory that must be empty.
    cache.mkdir(parents=True, exist_ok=True)
    for directory in (output, *output.parent.glob(output.name + ".old*")):
        for source in (directory / "cache" / "seedbs").glob("*.pkl"):
            target = cache / source.name
            if not target.exists():
                temporary = target.with_name(target.name + ".tmp")
                shutil.copyfile(source, temporary)
                temporary.replace(target)


def run_stage(name, run, args, context, train, progress, cache=None, force=False):
    output = ROOT / "runs" / f"{run}-{name}"
    if complete(name, output, context, train) and not force:
        print(f"[skip] {name}: {output}", flush=True)
        write_json(output / "resume.json", context)  # Adopt complete legacy outputs too.
        restore_progress(train["config"], progress, *STAGES[name])
        return False
    if cache is not None:
        import_legacy_cache(output, cache)
        args += ["--cache", cache]
    archive_incomplete(output)
    print(f"[run] {name}: {output}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), f"l2-{name}",
                    *map(str, args), "--output", str(output), "--progress", str(progress)], check=True, cwd=ROOT)
    if not complete(name, output, context, train):
        raise RuntimeError(f"Stage returned without complete outputs: {output}")
    write_json(output / "resume.json", context)
    return True


def run_pipeline(run, workers, progress, weights):
    if not run or run in (".", "..") or "/" in run or "\\" in run:
        raise ValueError("run-name must be a single directory name.")
    if workers < 1:
        raise ValueError("workers must be positive.")
    manifests = inputs_for(run)
    train = manifests["train"]
    train_dir, test_dir = (ROOT / "runs" / f"{run}-{side}-pos" for side in ("train", "test"))
    progress, weights = Path(progress).resolve(), Path(weights).resolve()
    cache_root = ROOT / "runs" / f"{run}-cache"
    context = {"inputs": {"train": train}, "dependencies": {}}
    mechanism_changed = run_stage("mechanism", run, ["--input", train_dir], context, train, progress)
    fit_changed = run_stage("fit", run, ["--input", train_dir, "--workers", workers], context, train, progress,
                            cache=cache_root / "fit-seedbs")
    fit_dir = ROOT / "runs" / f"{run}-fit"
    frozen = read_json(fit_dir / "frozen.json")
    check_frozen(frozen, train)
    mechanism = ROOT / "runs" / f"{run}-mechanism" / "mechanism.json"
    context = {"inputs": manifests, "dependencies": {"frozen": frozen,
               "fit_report": read_json(fit_dir / "fit_report.json"), "mechanism": read_json(mechanism)}}
    run_stage("test", run, ["--frozen", fit_dir, "--input", test_dir, "--workers", workers,
                           "--mechanism", mechanism], context, train, progress,
              cache=cache_root / "test-seedbs", force=mechanism_changed or fit_changed)
    context = {"inputs": manifests, "dependencies": {"frozen": frozen}}
    run_stage("cap", run, ["--frozen", fit_dir, "--train", train_dir, "--input", test_dir],
              context, train, progress, force=fit_changed)
    weights.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "l2-weights", "--progress", str(progress),
                    "--output", str(weights)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()
    run_pipeline(args.run, args.workers, args.progress, args.weights)


if __name__ == "__main__":
    main()
