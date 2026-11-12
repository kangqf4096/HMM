"""Eager safetensors loading for the project's pinned vLLM 0.8.x V0 engine.

0.8.x predates safetensors_load_strategy. Use its custom-loader extension;
keep the normal model initialization, FP8 postprocessing and shard filtering.
"""
from copy import copy
from time import perf_counter

from vllm.config import LoadFormat
from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import pt_weights_iterator


def eager_weights(files, use_tqdm):
    from safetensors.torch import load
    from tqdm.auto import tqdm
    for file in tqdm(files, desc='Loading eager safetensors shards', disable=not use_tqdm):
        with open(file, 'rb') as handle:
            data = handle.read()
        state = load(data)
        del data
        yield from state.items()
        del state


class EagerModelLoader(DefaultModelLoader):
    def __init__(self, load_config):
        config = copy(load_config)
        config.load_format = LoadFormat.AUTO
        super().__init__(config)

    def _get_weights_iterator(self, source):
        _, files, safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt,
            source.allow_patterns_overrides)
        iterator = (eager_weights if safetensors else pt_weights_iterator)(
            files, self.load_config.use_tqdm_on_load)
        if self.counter_before_loading_weights == 0:
            self.counter_before_loading_weights = perf_counter()
        return ((source.prefix + name, tensor) for name, tensor in iterator)
