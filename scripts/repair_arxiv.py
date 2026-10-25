"""Repair unreadable cached arXiv shards without removing model or other data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.arxiv_cache import repair_arxiv
from opt_hmm.io import read_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colab_arxiv.json")
    parser.add_argument("--report", default="tmp/arxiv-cache.json")
    args = parser.parse_args()
    repair_arxiv(read_json(args.config), args.report)
