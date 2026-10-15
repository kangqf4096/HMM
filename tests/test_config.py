import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opt_hmm.config import validate_text
from opt_hmm.io import read_json


class ConfigTests(unittest.TestCase):
    def test_shipped_text_configs_and_insertion_null_lengths(self):
        for filename in ("text_smoke.json", "text_pilot.json"):
            config = read_json(ROOT / "configs" / filename)
            lengths = validate_text(config)
            n = config["tokens"]
            self.assertEqual(lengths, [n - int(n * .1), n, n + int(n * .1)])

    def test_incompatible_generation_support_and_context_rejected(self):
        base = read_json(ROOT / "configs" / "text_smoke.json")
        for change in ({"temperature": .001}, {"prompt_tokens": 1}, {"tokens": 4},
                       {"max_model_len": 100}, {"switch_ranges": [[.1, .7], [.5, .9]]},
                       {"edits": [{"kind": "typo", "rate": .1}]}):
            config = copy.deepcopy(base)
            config.update(change)
            with self.assertRaises(ValueError):
                validate_text(config)


if __name__ == "__main__":
    unittest.main()
