"""CPU tests for the vLLM 0.8.4 V0 logits processor contract."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:  # The CPU test host may not have PyTorch.
    torch = None

from opt_hmm.protocol import PUBLIC_KEY, context_seed, gumbel_winner, uniforms
from opt_hmm.vllm_processor import (GumbelRequest, validate_sampling_params,
                                    validate_settings)

ROOT = Path(__file__).resolve().parents[1]


def settings(**overrides):
    value = {"context_width": 2, "key": PUBLIC_KEY, "vocab_size": 8,
             "blocked_ids": [0], "ranges": [[0, 2]],
             "trace_path": None, "pivot_diagnostics": False}
    value.update(overrides)
    return value


def sampling(**overrides):
    value = {"temperature": 1., "n": 1, "top_p": 1., "top_k": -1, "min_p": 0.,
             "repetition_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.,
             "min_tokens": 0, "logit_bias": None, "allowed_token_ids": None,
             "bad_words": [], "ignore_eos": True, "stop": []}
    value.update(overrides)
    return SimpleNamespace(**value)


class SettingsTest(unittest.TestCase):
    def test_valid_settings_include_empty_and_multi_ranges(self):
        validate_settings(settings(), 1.)
        validate_settings(settings(ranges=[]), .7)
        validate_settings(settings(ranges=[[0, 2], [4, 6]]), 1.)

    def test_invalid_settings_are_rejected(self):
        cases = [
            (None, 1.), (settings(), 0.), (settings(context_width=0), 1.),
            (settings(vocab_size=1), 1.), (settings(blocked_ids=[8]), 1.),
            (settings(blocked_ids=list(range(8))), 1.),
            (settings(ranges=[[1, 2], [0, 1]]), 1.),
            (settings(ranges=[[2, 2]]), 1.),
            (settings(key=""), 1.), (settings(key=b"key"), 1.),
            (settings(trace_path=3), 1.),
            (settings(pivot_diagnostics="yes"), 1.),
        ]
        for bad, temperature in cases:
            with self.subTest(bad=bad, temperature=temperature):
                with self.assertRaises(ValueError):
                    validate_settings(bad, temperature)

    def test_sampling_protocol_validation(self):
        validate_sampling_params(sampling())
        for bad in (sampling(temperature=0.), sampling(n=2), sampling(top_p=.9),
                    sampling(top_k=5), sampling(min_p=.1), sampling(repetition_penalty=1.1),
                    sampling(logit_bias={1: 1.}), sampling(allowed_token_ids=[1]),
                    sampling(bad_words=["x"]), sampling(ignore_eos=False),
                    sampling(stop=["x"])):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_sampling_params(bad)


@unittest.skipIf(torch is None, "torch is not installed")
class CallableTest(unittest.TestCase):
    def test_keyed_gumbel_choice_masking_history_and_trace(self):
        import numpy as np

        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="v0proc-") as folder:
            trace = Path(folder) / "audit.jsonl"
            current = settings(trace_path=str(trace))
            base = np.linspace(-2, 2, 8)
            suppressed = base.copy()
            suppressed[0] = -np.inf

            expected = gumbel_winner(suppressed, uniforms(context_seed([2, 3]), 8))
            first = GumbelRequest(current, 1.)([2, 3], [], torch.tensor(base, dtype=torch.float32))
            self.assertEqual(int(first.argmax()), expected)
            self.assertEqual(int(torch.isneginf(first).sum()), 7)

            expected2 = gumbel_winner(suppressed, uniforms(context_seed([3, expected]), 8))
            second = GumbelRequest(current, 1.)([2, 3], [expected], torch.tensor(base, dtype=torch.float32))
            self.assertEqual(int(second.argmax()), expected2)

            inactive = GumbelRequest(settings(ranges=[]), 1.)(
                [2, 3], [], torch.tensor(base, dtype=torch.float32))
            self.assertTrue(torch.isneginf(inactive[0]))
            self.assertTrue(np.allclose(inactive[1:].numpy(), base[1:]))

            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1])
            self.assertEqual(rows[0]["chosen"], expected)
            self.assertEqual(rows[1]["chosen"], expected2)


if __name__ == "__main__":
    unittest.main()
