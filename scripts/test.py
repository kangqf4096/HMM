"""Standard-library tests; numpy is the only required external dependency."""
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root / "tests")))
raise SystemExit(not result.wasSuccessful())
