"""Check every arXiv shard without downloading, repairing, or deleting data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.arxiv_check import check_arxiv
from opt_hmm.io import read_json
from opt_hmm.paths import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/colab_arxiv.json")
    parser.add_argument("--report", default="tmp/arxiv-check.json")
    parser.add_argument("--metadata-only", action="store_true", help="Quick header/footer and metadata checks; skip full decoding")
    parser.add_argument("--offline", action="store_true", help="Use the pinned 17-file manifest; skip remote file-size comparison")
    args = parser.parse_args()
    check_arxiv(read_json(project_path(args.config)), report_path=args.report,
                full=not args.metadata_only, offline=args.offline)


if __name__ == "__main__":
    main()
