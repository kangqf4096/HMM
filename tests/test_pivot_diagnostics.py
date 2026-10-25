import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from opt_hmm.detector import Detector, METHODS
from opt_hmm.emissions import Histogram
from opt_hmm.generate import _record
from opt_hmm.io import read_json, write_json, write_jsonl
from opt_hmm.pivot_diagnostics import (ConditionalHistogram, pivot_law,
    probabilities_from_logits, probability_summary)
from opt_hmm.pivot_experiment import build_ratio_methods, collect_rows, run
from opt_hmm.protocol import PUBLIC_KEY, context_seed, generation_contexts, gumbel_winner, uniforms


class DiagnosticMathTests(unittest.TestCase):
    def test_actual_mask_temperature_and_non_mutation(self):
        logits = np.array([-np.inf, -.8, .4, 1.3])
        original = logits.copy()
        p = probabilities_from_logits(logits, .7)
        self.assertEqual(p[0], 0)
        np.testing.assert_allclose(p[1:], np.exp(logits[1:] / .7) / np.exp(logits[1:] / .7).sum())
        summary = probability_summary(p)
        self.assertAlmostEqual(summary['s2'], (p*p).sum())
        u = uniforms(context_seed([1, 2]), 4)
        self.assertEqual(gumbel_winner(logits, u, .7), np.argmax(np.log(u[1:]) / p[1:]) + 1)
        np.testing.assert_array_equal(logits, original)

    def test_log_density_survives_extreme_tail(self):
        result = pivot_law(np.full(16, 1 / 16), 1e-100)
        self.assertAlmostEqual(result['log_pdf'], np.log(16) + 15 * np.log(1e-100))
        self.assertTrue(np.isfinite(result['log_pdf']))
        result = pivot_law([.5, .5], .3)
        self.assertAlmostEqual(result['cdf'], .09)
        self.assertAlmostEqual(np.exp(result['log_pdf']), .6)

    def test_external_ratios_preserve_all_existing_methods(self):
        y, mask = np.array([.1, .2, .4, .6, .8, .9]), np.array([0, 1, 1, 1, 1, 1], bool)
        density = Histogram(np.array([.2, .7, 1.3, 1.8]))
        for method in METHODS:
            detector = Detector(density, method=method)
            direct, supplied = detector.infer(y, mask), detector.infer_ratios(density(y), mask)
            np.testing.assert_allclose(direct.scores, supplied.scores)
            self.assertEqual((direct.epsilon, direct.rho), (supplied.epsilon, supplied.rho))
        self.assertFalse(Detector(density).infer_ratios([1, 1], [0, 0]).identifiable)

    def test_empty_s2_groups_fall_back_to_normalized_calibration(self):
        histogram = ConditionalHistogram([.7, .8, .9], [.5, .5, .5], bins=4)
        for density in histogram.densities:
            self.assertAlmostEqual(density.density.mean(), 1)
            self.assertTrue((density.density > 0).all())
        np.testing.assert_allclose(histogram.densities[0].density, histogram.global_density.density)


def toy_replay(prompt, tokens):
    return np.stack([np.array([.15, .25, .6]) if sum(prompt + tokens[:t]) % 2
                     else np.array([.55, .3, .15]) for t in range(len(tokens))])


class DiagnosticPipelineTests(unittest.TestCase):
    def fixture(self, path):
        config = read_json(ROOT / 'configs/text_smoke.json')
        config.update(tokens=10, context_width=2, prompt_tokens=2, bins=4, bootstrap=10)
        rng, sources, records = np.random.default_rng(7), [], []
        roles = [('calibration', 2), ('test', 2), ('null_cal', 20), ('null_test', 4)]
        for role, count in roles:
            for number in range(count):
                group, prompt, tokens, traces = f'{role}{number}', [1, 2], [], []
                active = role in ('calibration', 'test')
                mode = 'watermarked' if active else 'null'
                requested = 10 if active else 11
                for t in range(requested):
                    p = toy_replay(prompt, tokens + [0])[-1]
                    seed = context_seed((prompt + tokens)[-2:])
                    u = uniforms(seed, 3)
                    token = int(np.argmax(np.log(u) / p)) if active else int(rng.choice(3, p=p))
                    item = probability_summary(p)
                    if active:
                        item.update(pivot_law(p, u[token]))
                    traces.append(item)
                    tokens.append(token)
                contexts = generation_contexts(prompt, tokens, 2)
                source = {'id': f'{group}/{mode}', 'group': group, 'role': role,
                          'prompt_ids': prompt, 'token_ids': tokens, 'contexts': contexts,
                          'enabled': [active] * requested, 'seeds': [str(context_seed(c)) for c in contexts],
                          'probability_trace': traces}
                sources.append(source)
                doc = {'group': group, 'role': role}
                if active:
                    records.append(_record(doc, source, tokens, list(range(10)), mode, config, PUBLIC_KEY))
                else:
                    for length in (9, 10, 11):
                        records.append(_record(doc, source, tokens[:length], list(range(length)),
                                               f'null_n{length}', config, PUBLIC_KEY))
                if role == 'test':
                    changed, origin = tokens.copy(), list(range(10))
                    changed[4], origin[4] = (tokens[4] + 1) % 2, None
                    records.append(_record(doc, source, changed, origin, 'replace_0.1', config, PUBLIC_KEY))
                    records.append(_record(doc, source, tokens[:4] + [1] + tokens[4:],
                                           list(range(4)) + [None] + list(range(4, 10)),
                                           'insert_0.1', config, PUBLIC_KEY))
                    records.append(_record(doc, source, tokens[:4] + tokens[5:],
                                           list(range(4)) + list(range(5, 10)),
                                           'delete_0.1', config, PUBLIC_KEY))
        write_json(path / 'metadata.json', {'config': config, 'model': 'toy', 'pivot_diagnostics': True})
        write_jsonl(path / 'sources.jsonl', sources)
        write_jsonl(path / 'observations.jsonl', records)
        return config, sources, records

    def test_full_diagnostic_pipeline_and_independent_calibration(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='pd', dir=ROOT / 'tmp') as folder:
            input_dir, output = Path(folder) / 'in', Path(folder) / 'out'
            input_dir.mkdir()
            config, sources, records = self.fixture(input_dir)
            result = run(input_dir, output, replay=toy_replay)
            self.assertIn('source_alignment', result['methods'])
            thresholds = read_json(output / 'localization/thresholds.json')
            for method in ('model_context', 'source_alignment', 's2_histogram'):
                self.assertEqual(len(thresholds[method]), 3)
                self.assertTrue(all(t['count'] == 20 for t in thresholds[method].values()))
            report = read_json(output / 'report.json')
            self.assertLess(report['replay_agreement_with_actual_sampler']['s2_abs_difference']['max'], 1e-14)
            self.assertTrue(any(r['mean_tv'] > 0 for r in report['perturbations'] if r['variant'] == 'replace_0.1'))
            self.assertTrue((output / 'report.md').exists())
            # Flip all evaluation labels; densities and ratio construction must not change.
            source_map = {s['id']: s for s in sources}
            rows, _, _ = collect_rows(records, source_map, toy_replay)
            original_methods, _ = build_ratio_methods(rows, records, config)
            altered = copy.deepcopy(records)
            for record in altered:
                if record['role'] != 'calibration':
                    record['z'] = [not v for v in record['z']]
            altered_rows, _, _ = collect_rows(altered, source_map, toy_replay)
            altered_methods, _ = build_ratio_methods(altered_rows, altered, config)
            for method in original_methods:
                for ident in original_methods[method]:
                    np.testing.assert_array_equal(original_methods[method][ident], altered_methods[method][ident])


if __name__ == '__main__':
    unittest.main()
