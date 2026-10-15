"""Run from a source checkout, including before editable installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.cli import main

if __name__ == "__main__":
    main()
