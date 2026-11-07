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
        if kind not in ("insert", "delete", "replace", "resample") or not isinstance(rate, (float, int)) or not 0 <= rate < 1:
            raise ValueError("Edits require replace/insert/delete/resample and a rate in [0,1).")
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
    if not isinstance(config.get("enforce_eager", False), bool):
        raise ValueError("enforce_eager must be a boolean; eager mode is a runtime choice, not a string flag.")
    memory = config.get("gpu_memory_utilization", .8)
    if (isinstance(memory, bool) or not isinstance(memory, (int, float))
            or not isfinite(memory) or not 0 < memory <= 1):
        raise ValueError("gpu_memory_utilization must be a finite number in (0, 1].")
    if "max_num_batched_tokens" in config:
        batched = config["max_num_batched_tokens"]
        if type(batched) is not int or batched < max(config["max_model_len"], config["max_num_seqs"]):
            raise ValueError("max_num_batched_tokens must be an integer >= max_model_len and max_num_seqs.")
    return sorted(lengths)
