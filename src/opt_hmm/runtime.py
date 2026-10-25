"""Linux runtime and actual installed vLLM adapter contract checks."""
import platform
from importlib.metadata import version


def check_runtime():
    if platform.system() != "Linux":
        raise RuntimeError("vLLM generation is intended for the remote Linux host.")
    if version("vllm") != "0.19.1":
        raise RuntimeError("This adapter targets exactly vllm==0.19.1.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available.")
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "memory_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
               for i in range(torch.cuda.device_count())]
    return {"vllm": version("vllm"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "devices": devices}


def adapter_selftest():
    """Run against the real installed vLLM class, without loading an LLM."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
    from .protocol import PUBLIC_KEY, context_seed, gumbel_winner, uniforms
    from .vllm_processor import GumbelMaxProcessor

    def parameters(ranges):
        return SamplingParams(temperature=1., top_p=1., top_k=-1, ignore_eos=True,
                              extra_args={"opt_hmm": {"context_width": 2, "key": PUBLIC_KEY,
                              "vocab_size": 8, "blocked_ids": [0], "ranges": ranges}})
    marked, plain = parameters([[0, 8]]), parameters([])
    GumbelMaxProcessor.validate_params(marked)
    processor = GumbelMaxProcessor(SimpleNamespace(), torch.device("cpu"), False)
    live = []
    processor.update_state(BatchUpdate(2, [], [(0, marked, [2, 3], live), (1, plain, [4, 5], [])], []))
    base = np.linspace(-2, 2, 8)
    suppressed = base.copy()
    suppressed[0] = -np.inf
    expected = gumbel_winner(suppressed, uniforms(context_seed([2, 3]), 8))
    result = processor.apply(torch.tensor(np.tile(base, (2, 1)), dtype=torch.float32))
    assert int(result[0].argmax()) == expected
    assert torch.isneginf(result[0]).sum() == 7
    assert np.allclose(result[1, 1:].numpy(), base[1:])
    live.append(expected)
    processor.update_state(None)
    expected2 = gumbel_winner(suppressed, uniforms(context_seed([3, expected]), 8))
    result = processor.apply(torch.tensor(np.tile(base, (2, 1)), dtype=torch.float32))
    assert int(result[0].argmax()) == expected2, "Live output token list was not respected."
    processor.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
    result = processor.apply(torch.tensor(np.tile(base, (2, 1)), dtype=torch.float32))
    assert int(result[1].argmax()) == expected2
    assert np.allclose(result[0, 1:].numpy(), base[1:])
    processor.update_state(BatchUpdate(1, [1], [], []))
    assert len(processor.req_info) == 1
    # Verify that enabling telemetry reads the actual pre-sampling probabilities
    # and preserves the chosen token. Temp files stay within the repository.
    import json
    import tempfile
    from pathlib import Path
    from .vllm_processor import GumbelRequest
    temp_root = Path.cwd() / 'tmp'
    temp_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ad', dir=temp_root) as folder:
        trace_path = Path(folder) / 'a.jsonl'
        settings = dict(marked.extra_args['opt_hmm'])
        diagnostic_settings = dict(settings, pivot_diagnostics=True, trace_path=str(trace_path))
        first = GumbelRequest(settings, 1.)([2, 3], [], torch.tensor(base.copy()))
        second = GumbelRequest(diagnostic_settings, 1.)([2, 3], [], torch.tensor(base.copy()))
        assert torch.equal(first, second), 'Diagnostics changed watermark logits.'
        trace = json.loads(trace_path.read_text().strip())
        p = np.exp(suppressed - np.max(suppressed))
        p /= p.sum()
        assert np.isclose(trace['probability']['s2'], np.sum(p*p))
        assert trace['chosen'] == expected
    return {"adapter_selftest": "passed", "checks": ["argmax", "uniform support mask",
            "live token history", "batch swap", "request removal", "probability audit preserves sampling"]}
