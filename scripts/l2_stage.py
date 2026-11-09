"""Shared file handling for L2 stage recovery; no model or GPU dependencies."""
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def same_config(saved, current):
    ignored = {"model", "generation_storage"}
    return ({k: v for k, v in saved.items() if k not in ignored} ==
            {k: v for k, v in current.items() if k not in ignored})


def archive_incomplete(output):
    if not output.exists() or not any(output.iterdir()):
        return None
    index = 1
    backup = output.with_name(f"{output.name}.old{index}")
    while backup.exists():
        index += 1
        backup = output.with_name(f"{output.name}.old{index}")
    output.rename(backup)
    print(f"[restart] Preserved previous output: {backup}", flush=True)
    return backup


def restore_progress(config, progress, *stages):
    from opt_hmm.l2.stages import stage_table
    from opt_hmm.progress import Overall
    overall = Overall(stage_table(config), path=str(progress))
    overall.restore(progress)
    changed = False
    for stage in stages:
        remaining = overall.stages[stage]["total"] - overall.done[stage]
        if remaining:
            Path(progress).parent.mkdir(parents=True, exist_ok=True)
            overall.advance(stage, remaining, cached=True)
            changed = True
    if changed:
        overall.finish()
