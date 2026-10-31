"""vLLM 0.8.4 V0 callable logits processor for the watermark sampler.

vLLM 0.8.4's V0 engine applies each callable in
``SamplingParams.logits_processors`` to one sequence's next-token logits.  The
V1 engine does not support custom logits processors, so generation runs with
``VLLM_USE_V1=0``.  This module deliberately imports no vLLM symbols; it is safe
to import on the CPU-only test host.
"""
import json
from pathlib import Path

from .protocol import context_seed, uniforms
from .pivot_diagnostics import probabilities_from_logits, probability_summary, pivot_law


def validate_settings(settings, temperature):
    """Validate the per-request settings before generation starts."""
    if not isinstance(settings, dict):
        raise ValueError("opt_hmm settings must be a dictionary.")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
        raise ValueError("Protocol requires a positive sampling temperature.")
    context_width = settings.get("context_width")
    vocab = settings.get("vocab_size")
    if (not isinstance(context_width, int) or isinstance(context_width, bool)
            or context_width < 1):
        raise ValueError("Invalid context width.")
    if (not isinstance(vocab, int) or isinstance(vocab, bool) or vocab < 2):
        raise ValueError("Invalid vocabulary size.")
    key = settings.get("key", "")
    if not isinstance(key, str):
        raise ValueError("Watermark key must be a string.")
    context_seed([0] * context_width, key)
    blocked = settings.get("blocked_ids")
    if (not isinstance(blocked, list)
            or any(not isinstance(token, int) or isinstance(token, bool)
                   or not 0 <= token < vocab for token in blocked)
            or len(set(blocked)) >= vocab):
        raise ValueError("Invalid blocked token IDs.")
    ranges = settings.get("ranges")
    if not isinstance(ranges, list):
        raise ValueError("Watermark ranges must be a list.")
    previous = 0
    for bounds in ranges:
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(not isinstance(token, int) or isinstance(token, bool)
                       for token in bounds)
                or bounds[0] < previous or bounds[1] <= bounds[0]):
            raise ValueError("Ranges must be sorted nonoverlapping half-open intervals.")
        previous = bounds[1]
    trace_path = settings.get("trace_path")
    if trace_path is not None and not isinstance(trace_path, str):
        raise ValueError("trace_path must be a string or None.")
    diagnostics = settings.get("pivot_diagnostics", False)
    if not isinstance(diagnostics, bool):
        raise ValueError("pivot_diagnostics must be boolean.")


def validate_sampling_params(params):
    """Reject sampling settings that would break the watermark protocol."""
    if (params.temperature <= 0 or params.n != 1 or params.top_p != 1
            or params.top_k != -1 or params.min_p != 0
            or params.repetition_penalty != 1 or params.presence_penalty != 0
            or params.frequency_penalty != 0 or params.min_tokens != 0
            or params.logit_bias or params.allowed_token_ids or params.bad_words):
        raise ValueError("Protocol requires positive T, n=1, no truncation/penalties/logit bias.")
    if not params.ignore_eos or params.stop:
        raise ValueError("Protocol uses fixed lengths, ignore_eos=True, no text stop strings.")


class GumbelRequest:
    """Apply the keyed Gumbel-max watermark to one V0 sequence."""

    def __init__(self, settings, temperature):
        validate_settings(settings, temperature)
        self.settings = settings
        self.temperature = float(temperature)
        self.trace_path = settings.get("trace_path")

    def __call__(self, prompt_ids, output_ids, logits):
        import torch

        settings = self.settings
        vocab = settings["vocab_size"]
        if logits.dim() != 1:
            raise ValueError("Expected one logits row per sequence.")
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
        p = None
        if settings.get("pivot_diagnostics", False):
            # Read the actual masked pre-sampling logits, before forcing a winner.
            p = probabilities_from_logits(logits[:vocab].detach().double().cpu().numpy(),
                                          self.temperature)
            audit["probability"] = probability_summary(p)
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
            if p is not None:
                audit["probability"].update(pivot_law(p, float(u_np[chosen])))
                audit["probability"]["selected_probability"] = float(p[chosen])
            # Force the chosen winner, so vLLM does not add a second random draw.
            logits.fill_(float("-inf"))
            logits[chosen] = 0.
        if self.trace_path:
            with Path(self.trace_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit, separators=(",", ":")) + "\n")
        return logits
