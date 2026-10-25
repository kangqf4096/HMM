"""Download the configured OPT release into the project; run on Colab/Linux."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opt_hmm.progress import progress
from opt_hmm.paths import project_path, project_root, runtime_root


def download_model(config_path, model=None):
    root = project_root()
    config_path = project_path(config_path, root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repo, revision = config["model_repo"], config["model_revision"]
    if repo != "facebook/opt-1.3b":
        raise ValueError("This downloader selects the PyTorch files for facebook/opt-1.3b.")
    destination = project_path(model or config["model"], root)
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root / ".cache" / "hf")
    os.environ["HF_HUB_CACHE"] = str(root / ".cache" / "hf/hub")
    temp = runtime_root(root) / "tmp"
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp)
    progress("加载模型下载工具")
    from huggingface_hub import snapshot_download

    # One PyTorch checkpoint is sufficient. Do not also fetch TF/Flax weights.
    progress("下载 OPT-1.3B", detail="下载进度由 Hugging Face 报告")
    snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination),
                      allow_patterns=["*.json", "merges.txt", "pytorch_model.bin", "LICENSE.md"])
    progress("检查模型文件")
    required = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "pytorch_model.bin")
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Model download is incomplete; rerun this command. Missing: {missing}")
    source = {"repo_id": repo, "revision": revision,
              "url": f"https://huggingface.co/{repo}/tree/{revision}"}
    (destination / "opt_hmm_source.json").write_text(
        json.dumps(source, indent=2) + "\n", encoding="utf-8")
    progress("模型准备完成", 1, 1)
    return {"model": str(destination.resolve()), **source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colab_arxiv.json")
    parser.add_argument("--model", help="Override the local model directory")
    args = parser.parse_args()
    print(json.dumps(download_model(args.config, args.model), indent=2))


if __name__ == "__main__":
    main()
