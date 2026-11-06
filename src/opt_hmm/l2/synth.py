"""Toy-LM pipeline for L2: sources -> build -> mechanism -> fit -> test, no GPU. Never a result about OPT."""
from pathlib import Path
from time import perf_counter

import numpy as np

from ..io import new_output, provenance, write_json, write_jsonl
from ..l0.generate import source_record
from ..l0.synth import ToyLM
from ..paths import project_path
from ..protocol import PROTOCOL, context_seed, gumbel_winner, uniforms
from .build import build
from .config import validate_l2
from .plan import plan_jobs


def generate_sources(config, output, side, vocab=2000, seed=0):
    """Same sources.jsonl / metadata.json contract as l2.generate, from the toy LM."""
    validate_l2(config)
    spec = config["l2"]["sides"][side]
    key, n, m = spec["key"], config["tokens"], config["context_width"]
    lm = ToyLM(vocab, seed=seed)
    rng = np.random.default_rng([config["seed"], 99, 1 if side == "train" else 2])
    docs = []
    for role in spec["roles"]:
        for index in range(config["l2"]["roles"][role]["prompts"]):
            docs.append({"group": f"toy-{side}-{role}-{index}", "role": role,
                         "prompt_ids": rng.integers(0, vocab, config["prompt_tokens"]).tolist()})
    output = new_output(project_path(output))
    sources, started = [], perf_counter()
    for job in plan_jobs(docs, config):
        doc, ranges = job["doc"], job["ranges"]
        context, tokens, audit = list(doc["prompt_ids"]), [], {}
        for t in range(n):
            seed_t = context_seed(context[-m:], key)
            active = any(a <= t < b for a, b in ranges)
            row = {"step": t, "active": active, "context": context[-m:], "seed": str(seed_t)}
            if active:
                u = uniforms(seed_t, vocab)
                chosen = gumbel_winner(lm.logits(context), u)
                row.update({"chosen": chosen, "selected_y": float(u[chosen])})
            else:
                chosen = int(rng.choice(vocab, p=lm.probs(context)))
            tokens.append(chosen)
            context.append(chosen)
            audit[t] = row
        sources.append(source_record(job, tokens, audit, key, m))
    write_jsonl(output / "sources.jsonl", sources)
    write_json(output / "metadata.json", {
        "config": config, "model": "toy-lm", "toy": True, "protocol": PROTOCOL, "key": key, "side": side,
        "key_id": spec["key_id"], "vocab_size": vocab, "blocked_ids": [], "dtype": "float32", "toy_seed": seed,
        "generation_seconds": perf_counter() - started, "provenance": provenance(),
        "note": "Toy language model for pipeline tests only; not evidence about OPT."})
    return lm


def run_synth(config, output, vocab=2000, workers=1):
    """Whole L2 pipeline on the toy LM. Returns the paths of the reports."""
    from .evaluate import run_test
    from .fit import run_fit
    from .mechanism import run_mechanism
    base = Path(project_path(output))
    lm_train = generate_sources(config, base / "train-sources", "train", vocab)
    lm_test = generate_sources(config, base / "test-sources", "test", vocab)
    bar = base / "l2_progress.json"
    train = build(base / "train-sources", base / "train-pos", replay=lm_train.replay, progress_path=bar)
    test = build(base / "test-sources", base / "test-pos", replay=lm_test.replay, train_dir=base / "train-pos",
                 progress_path=bar)
    mech = run_mechanism(base / "train-pos", base / "mechanism", bar)
    fit = run_fit(base / "train-pos", base / "fit", workers=workers, progress_path=bar)
    result = run_test(base / "fit", base / "test-pos", base / "test", workers=workers,
                      mechanism=base / "mechanism" / "mechanism.json", progress_path=bar)
    return {"train": train, "test": test, "mechanism": mech, "fit": fit, "test_report": result}
