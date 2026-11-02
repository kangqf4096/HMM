"""Toy language model + the real watermark protocol: a CPU pipeline test, never a result.

The toy LM only supplies next-token distributions with varied entropy. Keys, seeds,
pivots, Gumbel-max sampling and all downstream L0 code are the real ones, so a toy
run exercises sources -> build -> evaluate end to end without a GPU.
"""
import hashlib
from time import perf_counter

import numpy as np

from ..io import new_output, provenance, write_json, write_jsonl
from ..paths import project_path
from ..protocol import PROTOCOL, PUBLIC_KEY, context_seed, gumbel_winner, uniforms
from .config import validate_l0
from .generate import source_record
from .schema import scenario_for

ROLE_ORDER = ("calibration", "dev", "test", "null_cal", "null_test")


class ToyLM:
    """P(. | last `order` tokens): Zipf base with a context-dependent sharpness plus noise."""

    def __init__(self, vocab=2000, order=3, seed=0):
        self.vocab, self.order = vocab, order
        rng = np.random.default_rng([seed, 1])
        self.log_rank = np.log1p(rng.permutation(vocab))
        self.seed = seed

    def _logits(self, context):
        digest = hashlib.blake2b(np.asarray(context[-self.order:], np.int64).tobytes(), digest_size=8,
                                 key=self.seed.to_bytes(8, "little")).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))
        sharpness = rng.uniform(0.3, 2.5)
        return -sharpness * self.log_rank + 0.5 * rng.standard_normal(self.vocab)

    def probs(self, context):
        logits = self._logits(context)
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def replay(self, prompt, tokens):
        """Distribution at every continuation position given the original prefix."""
        full = list(prompt) + list(tokens)
        return np.stack([self.probs(full[:len(prompt) + t]) for t in range(len(tokens))])

    def logits(self, context):
        return self._logits(context)


def generate_sources(config, output, vocab=2000, seed=0):
    """Same sources.jsonl / metadata.json contract as l0.generate, from the toy LM."""
    validate_l0(config)
    n, m, key = config["tokens"], config["context_width"], PUBLIC_KEY
    lm = ToyLM(vocab, seed=seed)
    rng = np.random.default_rng([config.get("seed", 20261015), 99])
    output = new_output(project_path(output))
    sources, started = [], perf_counter()
    for role in ROLE_ORDER:
        for index in range(config["roles"][role]):
            scenario = scenario_for(role, index)
            if scenario == "switch":
                ranges = [[int(a * n), int(b * n)] for a, b in config["switch_ranges"]]
            else:
                ranges = [] if scenario == "null" else [[0, n]]
            doc = {"group": f"toy-{role}-{index}", "role": role,
                   "prompt_ids": rng.integers(0, vocab, config["prompt_tokens"]).tolist()}
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
            job = {"doc": doc, "scenario": scenario, "ranges": ranges}
            sources.append(source_record(job, tokens, audit, key, m))
    write_jsonl(output / "sources.jsonl", sources)
    write_json(output / "metadata.json", {
        "config": config, "model": "toy-lm", "toy": True, "protocol": PROTOCOL, "key": key,
        "key_id": config.get("key_id"), "vocab_size": vocab, "blocked_ids": [], "dtype": "float32",
        "toy_seed": seed, "generation_seconds": perf_counter() - started, "provenance": provenance(),
        "note": "Toy language model for pipeline tests only; not evidence about OPT."})
    return lm, {"articles": len(sources), "seconds": perf_counter() - started}
