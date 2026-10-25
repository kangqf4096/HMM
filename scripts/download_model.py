"""Download the configured OPT release into the project; run on Colab/Linux."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_model(config_path, model=None):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo, revision = config["model_repo"], config["model_revision"]
    if repo != "facebook/opt-1.3b":
        raise ValueError("This downloader selects the PyTorch files for facebook/opt-1.3b.")
    destination = Path(model or config["model"])
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
    os.environ.setdefault("TMPDIR", str(ROOT / "tmp"))
    (ROOT / "tmp").mkdir(exist_ok=True)
    from huggingface_hub import snapshot_download

    # One PyTorch checkpoint is sufficient. Do not also fetch TF/Flax weights.
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination),
                      allow_patterns=["*.json", "merges.txt", "pytorch_model.bin", "LICENSE.md"])
    required = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "pytorch_model.bin")
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Model download is incomplete; rerun this command. Missing: {missing}")
    source = {"repo_id": repo, "revision": revision,
              "url": f"https://huggingface.co/{repo}/tree/{revision}"}
    (destination / "opt_hmm_source.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8")
    return {"model": str(destination.resolve()), **source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colab_arxiv.json")
    parser.add_argument("--model", help="Override the local model directory")
    args = parser.parse_args()
    print(json.dumps(download_model(args.config, args.model), indent=2))


if __name__ == "__main__":
    main()
