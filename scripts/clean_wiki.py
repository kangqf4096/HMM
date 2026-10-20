"""Remove this project's identified WikiText data before downloading arXiv."""
import json
import shutil
from pathlib import Path


def clean_wiki(root):
    root = Path(root).resolve()
    hf = root / ".cache" / "hf"
    targets = set()
    datasets = hf / "datasets"
    if datasets.is_dir():
        for path in datasets.iterdir():
            name = path.name.casefold()
            if (name == "wikitext" or name.endswith("___wikitext") or
                    ("wikitext" in name and name.endswith(".lock"))):
                targets.add(path)
    for parent in (hf / "hub", hf / "hub" / ".locks"):
        if parent.is_dir():
            targets.update(path for path in parent.iterdir()
                           if path.name.casefold().startswith("datasets--")
                           and path.name.casefold().endswith("--wikitext"))
    downloads = datasets / "downloads"
    if downloads.is_dir():
        for metadata in downloads.glob("*.json"):
            info = json.loads(metadata.read_text(encoding="utf-8"))
            if "wikitext" in str(info.get("url", "")).casefold():
                payload = metadata.with_name(metadata.name[:-5])
                targets.update(payload.with_name(payload.name + suffix)
                               for suffix in ("", ".json", ".lock", ".incomplete"))
        extracted = downloads / "extracted"
        if extracted.is_dir():
            targets.update(path for path in extracted.rglob("*")
                           if path.is_dir() and path.name.casefold() in
                           ("wikitext-2", "wikitext-2-raw", "wikitext-103", "wikitext-103-raw"))
    data = root / "data"
    if data.is_dir():
        for path in data.iterdir():
            manifest = path / "manifest.json"
            if not manifest.is_file():
                continue
            info = json.loads(manifest.read_text(encoding="utf-8"))
            source = info.get("dataset", info.get("config", {}).get("dataset", ""))
            if str(source).casefold() in ("wikitext", "salesforce/wikitext"):
                targets.add(path)

    # Validate all resolved paths before performing any recursive deletion.
    targets = sorted(path for path in targets if path.exists() or path.is_symlink())
    for path in targets:
        resolved = path.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise ValueError(f"Refusing to remove a path outside the project: {path}")
    removed = []
    for path in targets:
        if not path.exists() and not path.is_symlink():
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path.relative_to(root)))
    return removed


if __name__ == "__main__":
    removed = clean_wiki(Path(__file__).resolve().parents[1])
    for path in removed:
        print(f"Removed: {path}")
    print(f"WikiText cleanup complete: removed {len(removed)} entries.")
