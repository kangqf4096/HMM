import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opt_hmm.data import articles
from opt_hmm.edits import edit
from opt_hmm.protocol import (context_seed, extract_pivots, generation_contexts,
                              gumbel_winner, operational_labels, pivot, uniforms)


class ProtocolTests(unittest.TestCase):
    def test_scalar_jump_equals_full_vector(self):
        for context in ([0, 1, 2, 3], [151643, 5, 10, 20]):
            seed = context_seed(context)
            values = uniforms(seed, 50272)
            for token in (0, 1, 123, 50271):
                self.assertEqual(pivot(seed, token), values[token])
            self.assertTrue(((values > 0) & (values < 1)).all())

    def test_context_and_key_change_seed(self):
        seed = context_seed([1, 2])
        self.assertNotEqual(seed, context_seed([2, 1]))
        self.assertNotEqual(seed, context_seed([1, 2], "other"))

    def test_gumbel_formula_equivalence(self):
        rng = np.random.default_rng(77)
        for _ in range(100):
            logits, u, temperature = rng.normal(size=8), rng.uniform(.01, .99, 8), .7
            p = np.exp(logits / temperature)
            p /= p.sum()
            self.assertEqual(gumbel_winner(logits, u, temperature), np.argmax(np.log(u) / p))

    def test_repeated_context_repeats_pivot(self):
        y, mask, seeds = extract_pivots([1, 2, 3, 1, 2, 3], 2)
        self.assertEqual(y[2], y[5])
        self.assertEqual(seeds[2], seeds[5])
        np.testing.assert_array_equal(mask, [False, False, True, True, True, True])

    def test_replacement_changes_downstream_signal_and_then_recovers(self):
        original = [10, 11, 12, 13, 14, 15]
        contexts = generation_contexts([7, 8], original, 2)
        final, origin = [10, 11, 99, 13, 14, 15], [0, 1, None, 3, 4, 5]
        source, signal = operational_labels(final, origin, original, contexts, [1] * 6, 2)
        np.testing.assert_array_equal(source, [1, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(signal, [0, 0, 0, 0, 0, 1])

    def test_insertion_and_deletion_recovery(self):
        original = [10, 11, 12, 13, 14, 15]
        contexts = generation_contexts([7, 8], original, 2)
        _, signal = operational_labels([10, 11, 99, 12, 13, 14, 15],
                                       [0, 1, None, 2, 3, 4, 5],
                                       original, contexts, [1] * 6, 2)
        np.testing.assert_array_equal(signal, [0, 0, 0, 0, 0, 1, 1])
        _, signal = operational_labels([10, 11, 13, 14, 15], [0, 1, 3, 4, 5],
                                       original, contexts, [1] * 6, 2)
        np.testing.assert_array_equal(signal, [0, 0, 0, 0, 1])

    def test_edit_lengths_origin_maps_and_different_replacements(self):
        original, pool = list(range(100)), [10, 101, 102, 103]
        for kind, length in (("replace", 100), ("insert", 120), ("delete", 80)):
            final, origin, actions = edit(original, kind, .2, pool, np.random.default_rng(5))
            self.assertEqual(len(final), length)
            self.assertEqual(len(origin), length)
            self.assertEqual(len(actions), 20)
            for token, old in zip(final, origin):
                if old is not None:
                    self.assertEqual(token, original[old])
            if kind == "replace":
                self.assertEqual(sum(t != o for t, o in zip(final, original)), 20)

    def test_article_reconstruction_keeps_subheadings(self):
        rows = [{"text": t} for t in ["= First =\n", "paragraph", "", "= = Section = =", "body",
                                     "= Second =", "other"]]
        result = list(articles(rows, "train"))
        self.assertEqual([r["title"] for r in result], ["First", "Second"])
        self.assertIn("Section", result[0]["text"])
        self.assertIn("body", result[0]["text"])


if __name__ == "__main__":
    unittest.main()
