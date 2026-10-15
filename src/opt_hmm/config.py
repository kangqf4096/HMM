"""Reject incompatible protocol settings before expensive model generation."""
from math import isfinite


def validate_text(config):
    for name in ("tokens", "context_width", "prompt_tokens", "max_model_len", "max_num_seqs"):
        if not isinstance(config.get(name), int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if config["prompt_tokens"] < config["context_width"] or config["tokens"] <= config["context_width"]:
        raise ValueError("Prompt/continuation is too short for the watermark context.")
    temperature = config.get("temperature")
    if not isinstance(temperature, (float, int)) or not isfinite(temperature) or temperature < .01:
        raise ValueError("Use temperature >= 0.01; vLLM clamps smaller positive temperatures.")
    lengths, variants = {config["tokens"]}, set()
    for attack in config.get("edits", []):
        kind, rate = attack.get("kind"), attack.get("rate")
        if kind not in ("insert", "delete", "replace") or not isinstance(rate, (float, int)) or not 0 <= rate < 1:
            raise ValueError("Edits require replace/insert/delete and a rate in [0,1).")
        name = (kind, rate)
        if name in variants:
            raise ValueError("Duplicate edit condition.")
        variants.add(name)
        count = int(rate * config["tokens"])
        lengths.add(config["tokens"] + count if kind == "insert" else
                    config["tokens"] - count if kind == "delete" else config["tokens"])
    if min(lengths) <= config["context_width"]:
        raise ValueError("An edited sequence has no available pivots.")
    if config["prompt_tokens"] + max(lengths) > config["max_model_len"]:
        raise ValueError("Prompt plus continuation exceeds max_model_len.")
    previous = 0
    ranges = config.get("switch_ranges", [[.2, .7]])
    if not ranges:
        raise ValueError("At least one switch interval is required.")
    for bounds in ranges:
        if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 1:
            raise ValueError("Switch intervals must lie in [0,1].")
        start, stop = (int(v * config["tokens"]) for v in bounds)
        if start < previous or start == stop:
            raise ValueError("Switch intervals must be nonempty, ordered and nonoverlapping in tokens.")
        previous = stop
    return sorted(lengths)
