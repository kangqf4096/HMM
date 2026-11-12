"""Spill preserves active trace paths and subsequent append/commit behavior."""
import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opt_hmm.batch_cache import BatchCache
from opt_hmm.memory import cgroup_path


class PressureTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='sp-', dir=ROOT / 'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def cache(self):
        # Linux uses flock; Windows CPU tests exercise relocation with a local lock substitute.
        with patch('opt_hmm.batch_cache.cgroup_path', return_value=self.base), \
             patch('opt_hmm.batch_cache.threading.Thread.start'):
            cache = BatchCache(self.base / 'ram', 0, fallback_root=self.base / 'disk')
        cache.monitor = None
        cache.locked = contextlib.nullcontext
        cache.stage = lambda *args: cache._stage(*args)
        return cache

    def test_spill_preserves_trace_and_redirects_future_batches(self):
        probe = self.base / 'link'
        try:
            probe.symlink_to(self.base, target_is_directory=True)
            probe.unlink()
        except OSError:
            self.skipTest('Host does not permit symlinks; run this check on Linux')
        cache = self.cache()
        with patch('opt_hmm.batch_cache.memory_usage', return_value=0):
            first = cache.stage(0, [{'i': 1}], {}, 1)
            cache.stage(1, [{'i': 2}], {}, 1)
            trace = first / 'trace' / 'r0.jsonl'
            trace.write_text('first\n')
            cache.spill()
            self.assertTrue(first.is_symlink())
            self.assertEqual(trace.read_text(), 'first\n')
            with trace.open('a') as handle:
                handle.write('second\n')
            third = cache.stage(2, [{'i': 3}], {}, 1)
            self.assertTrue(third.is_symlink())
            for number in range(3):
                cache.commit(cache.root / f'b{number}', self.base / 'out' / f'b{number}')
        self.assertEqual((self.base / 'out/b0/trace/r0.jsonl').read_text(), 'first\nsecond\n')
        cache.finish()
        self.assertFalse(cache.root.exists())
        self.assertFalse(cache.fallback.exists())

    def test_failed_spill_keeps_original_data(self):
        cache = self.cache()
        with patch('opt_hmm.batch_cache.memory_usage', return_value=0):
            batch = cache.stage(0, [{'i': 1}], {}, 1)
        with patch('opt_hmm.batch_cache.shutil.copytree', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                cache.spill()
        self.assertTrue((batch / 'jobs.jsonl').exists())
        self.assertFalse(cache.spilled)

    def test_low_shm_or_memory_headroom_switches_before_writing(self):
        cache = self.cache()
        with patch('opt_hmm.batch_cache.memory_usage', return_value=90_000_000_000), \
             patch.object(cache, '_spill_locked') as spill:
            # Keep this Windows-compatible: no symlink is needed to check triggering.
            cache.stage(0, [], {}, 1)
        spill.assert_called_once()

    def test_locates_nested_cgroup_mount(self):
        files = {'/proc/self/cgroup': '0::/jobs/a\n',
                 '/proc/self/mountinfo': '1 0 0:1 /jobs /sys/fs/cgroup rw - cgroup2 cgroup rw\n'}
        with patch.object(Path, 'read_text', lambda path: files[str(path).replace('\\', '/')]), \
             patch.object(Path, 'is_file', return_value=True):
            self.assertEqual(cgroup_path().as_posix(), '/sys/fs/cgroup/a')


class EagerTests(unittest.TestCase):
    def test_eager_reads_bytes_and_retains_prefix_and_fp8_loader_parent(self):
        from enum import Enum
        class Format(Enum):
            AUTO = 'auto'

        class Default:
            def __init__(self, config):
                self.load_config = config
                self.counter_before_loading_weights = 0

            def _prepare_weights(self, *args):
                return '', [self.load_config.file], True

        seen = []
        def load(data):
            seen.append(data)
            return {'weight': [1, 2, 3]}

        modules = {
            'vllm.config': types.SimpleNamespace(LoadFormat=Format),
            'vllm.model_executor.model_loader.loader': types.SimpleNamespace(DefaultModelLoader=Default),
            'vllm.model_executor.model_loader.weight_utils': types.SimpleNamespace(pt_weights_iterator=None),
            'safetensors.torch': types.SimpleNamespace(load=load),
            'tqdm.auto': types.SimpleNamespace(tqdm=lambda files, **kw: files)}
        with tempfile.TemporaryDirectory(prefix='eg-', dir=ROOT / 'tmp') as folder:
            file = Path(folder) / 'w.safetensors'
            file.write_bytes(b'checkpoint')
            spec = importlib.util.spec_from_file_location('eg_test', ROOT / 'src/opt_hmm/eager_loader.py')
            mod = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, modules):
                spec.loader.exec_module(mod)
                original = types.SimpleNamespace(file=str(file), use_tqdm_on_load=False, load_format='custom')
                loader = mod.EagerModelLoader(original)
                source = types.SimpleNamespace(model_or_path='', revision=None, fall_back_to_pt=True,
                                               allow_patterns_overrides=None, prefix='fp8.')
                self.assertEqual(list(loader._get_weights_iterator(source)), [('fp8.weight', [1, 2, 3])])
                self.assertEqual(seen, [b'checkpoint'])
                self.assertEqual(original.load_format, 'custom')
                self.assertEqual(loader.load_config.load_format, Format.AUTO)


class SmokeGuardTests(unittest.TestCase):
    def test_existing_hard_limit_needs_no_cgroup_write(self):
        spec = importlib.util.spec_from_file_location('smoke_test', ROOT / 'scripts/cache_smoke.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory(prefix='cg-', dir=ROOT / 'tmp') as folder:
            group = Path(folder)
            (group / 'memory.max').write_text('100000000000')
            with patch.object(mod, 'cgroup_path', return_value=group):
                self.assertEqual(mod.limited_group(), (group, False))
            self.assertEqual(list(group.iterdir()), [group / 'memory.max'])

    def test_no_delegation_fails_before_model_load_and_keeps_parent_limit(self):
        spec = importlib.util.spec_from_file_location('smoke_test', ROOT / 'scripts/cache_smoke.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory(prefix='cg-', dir=ROOT / 'tmp') as folder:
            group = Path(folder)
            (group / 'memory.max').write_text('max')
            with patch.object(mod, 'cgroup_path', return_value=group):
                with self.assertRaisesRegex(RuntimeError, 'Cannot enforce 100 GB'):
                    mod.limited_group()
            self.assertEqual((group / 'memory.max').read_text(), 'max')
            self.assertEqual(list(group.iterdir()), [group / 'memory.max'])
