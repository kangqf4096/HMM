"""Linux runtime and vLLM 0.8.4 V0 contract checks."""
import json
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# The project relies on V0's per-request callable logits processors.  Set the
# engine choice before vLLM is imported anywhere in this process.
os.environ.setdefault("VLLM_USE_V1", "0")

SUPPORTED_VLLM = "0.8.4"


def _installed_vllm():
    try:
        return version("vllm")
    except PackageNotFoundError as error:
        raise RuntimeError("vLLM 0.8.4 is not installed; use the server image or scripts/install.sh.") from error


def check_runtime():
    if platform.system() != "Linux":
        raise RuntimeError("vLLM generation is intended for the remote Linux host.")
    installed = _installed_vllm()
    if installed.split("+", 1)[0] != SUPPORTED_VLLM:
        raise RuntimeError(f"This adapter targets vllm=={SUPPORTED_VLLM}; found {installed}.")
    if os.environ.get("VLLM_USE_V1", "0").strip().lower() not in ("0", "false"):
        raise RuntimeError("VLLM_USE_V1 must be 0: vLLM 0.8.4 V1 ignores custom logits processors.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available.")
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "memory_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
               for i in range(torch.cuda.device_count())]
    return {"vllm": installed, "engine": "V0", "torch": torch.__version__,
            "cuda": torch.version.cuda, "devices": devices}


def adapter_selftest():
    """Exercise the installed V0 callable contract without loading an LLM."""
    import numpy as np
    import torch

    from .protocol import PUBLIC_KEY, context_seed, gumbel_winner, uniforms
    from .vllm_processor import GumbelRequest, validate_sampling_params, validate_settings

    from vllm import SamplingParams
    probe = SamplingParams(n=1, temperature=1., top_p=1., top_k=-1, min_p=0.,
                           max_tokens=1, ignore_eos=True, detokenize=False,
                           logits_processors=[GumbelRequest({"context_width": 2,
                               "key": PUBLIC_KEY, "vocab_size": 8, "blocked_ids": [0],
                               "ranges": [[0, 1]], "trace_path": None,
                               "pivot_diagnostics": False}, 1.)])
    validate_sampling_params(probe)
    assert probe.logits_processors, "SamplingParams.logits_processors is not retained."

    temp_root = Path.cwd() / "tmp"
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v0-adapter-", dir=temp_root) as folder:
        trace_path = Path(folder) / "audit.jsonl"
        settings = {"context_width": 2, "key": PUBLIC_KEY, "vocab_size": 8,
                    "blocked_ids": [0], "ranges": [[0, 2]],
                    "trace_path": str(trace_path), "pivot_diagnostics": False}
        validate_settings(settings, 1.)
        base = np.linspace(-2, 2, 8)
        suppressed = base.copy()
        suppressed[0] = -np.inf

        expected = gumbel_winner(suppressed, uniforms(context_seed([2, 3]), 8))
        result = GumbelRequest(settings, 1.)([2, 3], [], torch.tensor(base, dtype=torch.float32))
        assert int(result.argmax()) == expected
        assert int(torch.isneginf(result).sum()) == 7

        inactive = dict(settings, ranges=[])
        result = GumbelRequest(inactive, 1.)([2, 3], [], torch.tensor(base, dtype=torch.float32))
        assert int(torch.isneginf(result[0])) == 1
        assert np.allclose(result[1:].numpy(), base[1:])

        expected2 = gumbel_winner(suppressed, uniforms(context_seed([3, expected]), 8))
        result = GumbelRequest(settings, 1.)([2, 3], [expected], torch.tensor(base, dtype=torch.float32))
        assert int(result.argmax()) == expected2

        expected_hot = gumbel_winner(suppressed, uniforms(context_seed([2, 3]), 8), 2.)
        result = GumbelRequest(settings, 2.)([2, 3], [], torch.tensor(base, dtype=torch.float32))
        assert int(result.argmax()) == expected_hot

        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 4
        assert [row["step"] for row in rows] == [0, 0, 1, 0]
        assert rows[0]["chosen"] == expected and rows[2]["chosen"] == expected2
        assert rows[1]["active"] is False and rows[3]["chosen"] == expected_hot

        for bad in (dict(settings, context_width=0), dict(settings, vocab_size=1),
                    dict(settings, blocked_ids=[8]), dict(settings, ranges=[[1, 1]]),
                    dict(settings, key=""), dict(settings, pivot_diagnostics="yes")):
            try:
                validate_settings(bad, 1.)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid settings were accepted: {bad}")
    return {"adapter_selftest": "passed", "checks": [
        "SamplingParams.logits_processors field", "V0 callable signature",
        "blocked-token mask", "inactive range", "live token history",
        "temperature scaling", "trace audit", "settings validation"]}
