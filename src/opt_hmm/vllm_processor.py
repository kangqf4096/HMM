"""vLLM 0.19.1 V1 adapter. Imported only in the Linux inference environment."""
import json
from pathlib import Path

import numpy as np
import torch
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .protocol import context_seed, uniforms


class GumbelRequest:
    def __init__(self, settings, temperature):
        self.settings = settings
        self.temperature = temperature
        self.trace_path = settings.get("trace_path")

    def __call__(self, prompt_ids, output_ids, logits):
        settings = self.settings
        vocab = settings["vocab_size"]
        if logits.numel() < vocab:
            raise ValueError("Logits are smaller than the frozen protocol vocabulary.")
        # Same support restriction for marked and unmarked generation.
        logits[vocab:] = float("-inf")
        blocked = settings["blocked_ids"]
        if blocked:
            logits[torch.tensor(blocked, device=logits.device, dtype=torch.long)] = float("-inf")
        step = len(output_ids)
        context = (list(prompt_ids) + list(output_ids))[-settings["context_width"]:]
        if len(context) != settings["context_width"]:
            raise ValueError("Missing generation context.")
        seed = context_seed(context, settings["key"])
        active = any(start <= step < stop for start, stop in settings["ranges"])
        audit = {"step": step, "active": active, "context": context, "seed": str(seed)}
        if active:
            # l/T + standard Gumbel noise has exactly the same argmax as
            # log(U_w)/P_w, P=softmax(l/T), and avoids dividing by tiny P_w.
            u_np = uniforms(seed, vocab)
            u = torch.from_numpy(u_np).to(device=logits.device)
            scores = logits[:vocab].double() / self.temperature - torch.log(-torch.log(u))
            if not torch.isfinite(scores).any():
                raise ValueError("No finite token available.")
            chosen = int(torch.argmax(scores).item())
            audit.update({"chosen": chosen, "selected_y": float(u_np[chosen])})
            # Force the chosen winner, so vLLM does not add a second random draw.
            logits.fill_(float("-inf"))
            logits[chosen] = 0.
        if self.trace_path:
            with Path(self.trace_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, separators=(",", ":")) + "\n")
        return logits


class GumbelMaxProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        settings = (params.extra_args or {}).get("opt_hmm")
        if settings is None:
            return
        if not isinstance(settings, dict):
            raise ValueError("opt_hmm settings must be a dictionary.")
        m, vocab = settings.get("context_width"), settings.get("vocab_size")
        if not isinstance(m, int) or m < 1 or not isinstance(vocab, int) or vocab < 2:
            raise ValueError("Invalid context width or vocabulary.")
        context_seed([0] * m, settings.get("key", ""))
        blocked = settings.get("blocked_ids")
        if (not isinstance(blocked, list) or any(not isinstance(t, int) or not 0 <= t < vocab for t in blocked)
                or len(set(blocked)) >= vocab):
            raise ValueError("Invalid blocked token IDs.")
        ranges = settings.get("ranges")
        if not isinstance(ranges, list):
            raise ValueError("Watermark ranges must be a list.")
        previous = 0
        for bounds in ranges:
            if (len(bounds) != 2 or any(not isinstance(t, int) for t in bounds)
                    or bounds[0] < previous or bounds[1] <= bounds[0]):
                raise ValueError("Ranges must be sorted nonoverlapping half-open intervals.")
            previous = bounds[1]
        if (params.temperature <= 0 or params.n != 1 or params.top_p != 1
                or params.top_k != -1 or params.min_p != 0
                or params.repetition_penalty != 1 or params.presence_penalty != 0
                or params.frequency_penalty != 0 or params.min_tokens != 0
                or params.logit_bias or params.allowed_token_ids or params.bad_words):
            raise ValueError("Protocol requires positive T, n=1, no truncation/penalties/logit bias.")
        if not params.ignore_eos or params.stop:
            raise ValueError("Protocol uses fixed lengths, ignore_eos=True, no text stop strings.")

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        settings = (params.extra_args or {}).get("opt_hmm")
        return None if settings is None else GumbelRequest(settings, params.temperature)
