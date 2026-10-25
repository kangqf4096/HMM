"""Real-model replay diagnostics and independently calibrated emission comparisons.

The source-alignment method is an experimental oracle using original prompts and
origin alignment, but never uses Z to choose a density. All replay methods have
access to the original prompt and are not prompt-free deployment detectors.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiment import run_evaluation
from .io import new_output, read_json, read_jsonl, write_json, write_jsonl, provenance
from .pivot_diagnostics import (ConditionalHistogram, S2_EDGES, law_ratio, pivot_law,
                                probabilities_from_logits, probability_summary,
                                source_for_record, uniform_distance)


class ModelReplay:
    def __init__(self, model_path, metadata, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Probability replay requires a CUDA GPU; run on the Linux server.')
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[metadata['dtype']]
        self.torch, self.device = torch, device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                        torch_dtype=dtype).to(device).eval()
        self.metadata = metadata

    def __call__(self, prompt, tokens):
        torch = self.torch
        full = list(prompt) + list(tokens)
        if len(full) > self.metadata['config']['max_model_len']:
            raise ValueError('Replay exceeds frozen max_model_len.')
        with torch.inference_mode():
            inputs = torch.tensor([full], device=self.device, dtype=torch.long)
            result = self.model(input_ids=inputs, use_cache=False)
            # Logit at the last prefix position predicts the next token.
            logits = result.logits[0, len(prompt)-1:len(full)-1, :self.metadata['vocab_size']].float().cpu().numpy()
            del result, inputs
        logits[:, self.metadata['blocked_ids']] = -np.inf
        return np.stack([probabilities_from_logits(row, self.metadata['config']['temperature']) for row in logits])


def collect_rows(records, sources, replay):
    """Build probability-based inputs without reading any truth labels."""
    grouped = defaultdict(list)
    for record in records:
        grouped[source_for_record(record, sources)['id']].append(record)
    rows, audit, totals = [], [], {'ratio_log_floor': 0, 'source_fallback_positions': 0}
    for number, (source_id, selected) in enumerate(grouped.items(), 1):
        source = sources[source_id]
        print(f'Probability replay {number}/{len(grouped)}: {source_id}', flush=True)
        original_p = replay(source['prompt_ids'], source['token_ids'])
        original_summaries = [probability_summary(p) for p in original_p]
        trace = source.get('probability_trace')
        if not trace or any(item is None for item in trace):
            raise ValueError('Missing pre-sampling probability audit. Regenerate with --pivot-diagnostics.')
        for record in selected:
            tokens = record['token_ids']
            is_original_prefix = tokens == source['token_ids'][:len(tokens)]
            current_p = original_p[:len(tokens)] if is_original_prefix else replay(source['prompt_ids'], tokens)
            for t, valid in enumerate(record['mask']):
                if not valid:
                    continue
                y, old = float(record['y'][t]), record['origin'][t]
                p_eval = current_p[t]
                summary = original_summaries[t] if is_original_prefix else probability_summary(p_eval)
                eval_law = pivot_law(p_eval, y)
                p_source = original_p[old] if old is not None else p_eval
                source_law = pivot_law(p_source, y)
                totals['ratio_log_floor'] += int(eval_law['log_pdf'] < -700) + int(source_law['log_pdf'] < -700)
                totals['source_fallback_positions'] += int(old is None)
                preserved = old is not None and record['verification_seeds'][t] == source['seeds'][old]
                row = {'id': record['id'], 'group': record['group'], 'role': record['role'],
                       'variant': record['variant'], 't': t, 'y': y, **summary,
                       'eval_cdf': eval_law['cdf'], 'eval_log_pdf': eval_law['log_pdf'],
                       'source_log_pdf': source_law['log_pdf'], 'source_cdf': source_law['cdf'],
                       'source_fallback': old is None, 'seed_preserved': preserved,
                       'source_s2': original_summaries[old]['s2'] if old is not None else None,
                       'probability_tv': float(np.abs(p_source - p_eval).sum() / 2) if old is not None else None,
                       'cdf_difference_at_y': abs(source_law['cdf'] - eval_law['cdf']) if old is not None else None}
                rows.append(row)
                if is_original_prefix and trace[t].get('cdf') is not None:
                    audit.append({'group': record['group'], 'source': source_id, 't': t,
                                  's2_abs_difference': abs(summary['s2'] - trace[t]['s2']),
                                  'pmax_abs_difference': abs(summary['pmax'] - trace[t]['pmax']),
                                  'cdf_abs_difference': abs(eval_law['cdf'] - trace[t]['cdf']),
                                  'log_pdf_abs_difference': abs(eval_law['log_pdf'] - trace[t]['log_pdf'])})
            if not is_original_prefix:
                del current_p
        del original_p
    return rows, audit, totals


def build_ratio_methods(rows, records, config):
    cal = [r for r in rows if r['role'] == 'calibration']
    histogram = ConditionalHistogram([r['y'] for r in cal], [r['s2'] for r in cal],
                                     bins=config['bins'], pseudocount=config['pseudocount'])
    methods = {name: {r['id']: np.ones(len(r['y'])) for r in records}
               for name in ('model_context', 'source_alignment', 's2_histogram')}
    for row in rows:
        ident, t = row['id'], row['t']
        methods['model_context'][ident][t] = law_ratio(row['eval_log_pdf'])
        methods['source_alignment'][ident][t] = law_ratio(row['source_log_pdf'])
        methods['s2_histogram'][ident][t] = histogram(row['y'], row['s2'])
    return methods, histogram


def _mean(values):
    values = [x for x in values if x is not None]
    return float(np.mean(values)) if values else None


def summarize(rows, audit, sources, records, config):
    from .emissions import Histogram
    by_id = {r['id']: r for r in records}
    cal = [r['y'] for r in rows if r['role'] == 'calibration']
    pooled = Histogram.fit(cal, config['bins'], config['pseudocount'])
    selected_groups = defaultdict(list)
    for row in rows:
        # Labels are used only for diagnostic summaries, never ratio construction.
        if by_id[row['id']]['z'][row['t']]:
            selected_groups[(row['role'], row['variant'])].append(row)
    shifts = []
    for (role, variant), selected in sorted(selected_groups.items()):
        shifts.append({'role': role, 'variant': variant, 'positions': len(selected),
                      'documents': len({r['group'] for r in selected}),
                      'mean_source_s2': _mean(r['source_s2'] for r in selected),
                      'mean_eval_s2': _mean(r['s2'] for r in selected),
                      'mean_x': _mean(-np.log(r['y']) for r in selected),
                      'eval_pit_uniform_distance': uniform_distance([r['eval_cdf'] for r in selected]),
                      'mean_eval_log_score_minus_pooled': _mean(r['eval_log_pdf'] - np.log(pooled([r['y']])[0]) for r in selected)})
    perturbations = []
    for variant in sorted({r['variant'] for r in rows}):
        for preserved in (False, True):
            selected = [r for r in rows if r['variant'] == variant and r['seed_preserved'] == preserved
                        and r['probability_tv'] is not None and r['role'] == 'test']
            if selected:
                perturbations.append({'variant': variant, 'seed_preserved': preserved, 'positions': len(selected),
                                      'mean_tv': _mean(r['probability_tv'] for r in selected),
                                      'max_tv': max(r['probability_tv'] for r in selected),
                                      'mean_cdf_difference_at_y': _mean(r['cdf_difference_at_y'] for r in selected)})
    generation_groups, document_groups = defaultdict(list), defaultdict(list)
    m = config['context_width']
    for source in sources.values():
        if source['role'] not in ('calibration', 'dev', 'test'):
            continue
        trace = source['probability_trace']
        for t in range(m, len(trace)):
            if not source['enabled'][t]:
                continue
            p = trace[t]
            group = min(int(np.searchsorted(S2_EDGES, p['s2'], side='right') - 1), len(S2_EDGES)-2)
            item = {**p, 'x': -np.log(float(source['pivots'][t]))}
            generation_groups[(source['role'], group)].append(item)
            document_groups[(source['id'], group)].append(item)
    generation = []
    for (role, group), selected in sorted(generation_groups.items()):
        generation.append({'role': role, 's2_interval': S2_EDGES[group:group+2].tolist(),
                           'positions': len(selected), 'mean_x': _mean(r['x'] for r in selected),
                           'predicted_mean_x': _mean(r['s2'] for r in selected),
                           'pit_uniform_distance': uniform_distance([r['cdf'] for r in selected])})
    documents = [{'source': source_id, 's2_group': group, 'positions': len(items),
                  'mean_residual_x_minus_s2': _mean(r['x'] - r['s2'] for r in items)}
                 for (source_id, group), items in sorted(document_groups.items())]
    agreement = {name: {'mean': _mean(r[name] for r in audit),
                        'max': max((r[name] for r in audit), default=None)}
                 for name in ('s2_abs_difference', 'pmax_abs_difference', 'cdf_abs_difference', 'log_pdf_abs_difference')}
    return {'generation_moments': generation, 'generation_document_residuals': documents,
            'replay_agreement_with_actual_sampler': agreement,
            'surviving_signal_by_condition': shifts, 'perturbations': perturbations,
            'diagnostic_note': 'KS distances are descriptive; no IID token-level p-values. Compare held-out test roles.'}


def run(input_dir, output, model_path=None, device='cuda', replay=None):
    input_dir = Path(input_dir)
    metadata = read_json(input_dir / 'metadata.json')
    if not metadata.get('pivot_diagnostics'):
        raise ValueError('Regenerate with --pivot-diagnostics to capture actual pre-sampling probabilities.')
    model_path = model_path or metadata['model']
    if replay is None and not Path(model_path).is_dir():
        raise ValueError(f'Model directory does not exist: {model_path}')
    output = new_output(output)
    config = metadata['config']
    records = read_jsonl(input_dir / 'observations.jsonl')
    sources = {r['id']: r for r in read_jsonl(input_dir / 'sources.jsonl')}
    for source in sources.values():
        trace = source.get('probability_trace')
        if not trace or len(trace) != len(source['token_ids']) or any(p is None for p in trace):
            raise ValueError('Incomplete probability traces. Regenerate with --pivot-diagnostics.')
    # Recover generation pivots from the already-audited source token and seed.
    from .protocol import pivot
    for source in sources.values():
        source['pivots'] = [pivot(int(seed), token) for seed, token in zip(source['seeds'], source['token_ids'])]
    started = perf_counter()
    replay = replay or ModelReplay(model_path, metadata, device)
    rows, audit, numerical = collect_rows(records, sources, replay)
    methods, histogram = build_ratio_methods(rows, records, config)
    write_jsonl(output / 'positions.jsonl', rows)
    write_jsonl(output / 'replay_agreement.jsonl', audit)
    write_json(output / 'conditional_density.json', histogram.to_dict())
    report = summarize(rows, audit, sources, records, config)
    report.update({'config': config, 'provenance': provenance(), 'numerical': numerical,
                   'model': model_path, 'replay_device': device, 'replay_seconds': perf_counter() - started,
                   'method_information': {'model_context': 'HF replay on final tokens with original prompt',
                    'source_alignment': 'Original-history replay for all retained tokens, without consulting Z; final-history fallback for inserted/replaced tokens',
                    's2_histogram': 'Fixed S2 groups, fitted only to independent pure calibration documents'},
                   'limitations': ['HF replay can differ from actual vLLM logits; inspect replay agreement.',
                    'All three additional methods use original prompts; source_alignment also uses true origin alignment.',
                    'Small existing smoke/pilot sets provide exploratory diagnostics, not precise FPR or generalization claims.']})
    evaluation = run_evaluation(records, new_output(output / 'localization'), config, ratio_methods=methods)
    report['seconds'] = perf_counter() - started
    write_json(output / 'report.json', report)
    lines = ['# Real-model pivot diagnostics', '', 'Read alongside localization/report.md and report.json.', '',
             'All added methods use original prompts. Source alignment is an evaluation oracle.', '',
             '## Actual sampler: E[-log Y | P] = S2(P)', '',
             '| Role | S2 interval | Positions | Mean X | Predicted mean X | PIT KS distance |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['generation_moments']:
        lines.append(f"| {row['role']} | {row['s2_interval']} | {row['positions']} | {row['mean_x']:.5f} | {row['predicted_mean_x']:.5f} | {row['pit_uniform_distance']:.5f} |")
    lines += ['', '## Original versus edited history (test documents)', '',
              '| Variant | Seed preserved | Positions | Mean TV(Pgen, Peval) | Max TV | Mean CDF difference at Y |',
              '|---|---|---:|---:|---:|---:|']
    for row in report['perturbations']:
        lines.append(f"| {row['variant']} | {row['seed_preserved']} | {row['positions']} | {row['mean_tv']:.5f} | {row['max_tv']:.5f} | {row['mean_cdf_difference_at_y']:.5f} |")
    lines += ['', '## Replay accuracy', '', 'HF replay is an approximation to vLLM; check these errors before interpreting conditional emission comparisons.', '']
    for name, values in report['replay_agreement_with_actual_sampler'].items():
        lines.append(f'- {name}: mean={values["mean"]}, max={values["max"]}')
    lines += ['', 'No IID token-level p-values. Document-group residuals and full per-position diagnostics are in report.json/positions.jsonl.',
              'Each added localization method reruns the complete independent null calibration at the original length/mask.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return {'report': str(output / 'report.md'), 'localization_report': str(output / 'localization/report.md'),
            'positions': len(rows), 'seconds': report['seconds'], 'methods': list(evaluation['summary'])}
