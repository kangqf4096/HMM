"""Run real L2 generation under a 100 GB cgroup ceiling; keep smoke artifacts."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from opt_hmm.memory import CAP_BYTES, cgroup_path, memory_usage


def worker(args):
    if args.group:
        (Path(args.group) / 'cgroup.procs').write_text(str(os.getpid()))
    from opt_hmm.io import read_json
    from opt_hmm.l0.generate import generate
    from opt_hmm.l2.config import validate_l2
    from opt_hmm.l2.plan import plan_jobs
    config = read_json(ROOT / args.config)
    limit_bytes = int((cgroup_path() / 'memory.max').read_text())
    config['generation_storage'] = {'cache_root': '/dev/shm/cache',
                                    'fallback_root': '/root/blockdata/cache',
                                    'reserve_gb': 1, 'spill_gb': min(args.spill_gb, limit_bytes * .9 / 1e9)}
    chunk = config['l2'].get('generate_chunk', 200)
    limit = args.jobs or chunk + 1  # Fill one production batch, then exercise rollover.
    spec = config['l2']['sides']['train']

    def planner(prompts, cfg):
        jobs = plan_jobs([d for d in prompts if d['role'] in spec['roles']], cfg)
        if len(jobs) < limit:
            raise ValueError(f'Need {limit} train jobs, found {len(jobs)} in prompts.')
        return jobs[:limit]

    result = generate(config, args.model, args.prompts, args.output / 'gen',
                      gpu_memory=config.get('gpu_memory_utilization', .8),
                      dtype=config.get('dtype', 'bfloat16'), validate=validate_l2,
                      planner=planner, key=spec['key'], key_id=spec['key_id'], chunk=chunk)
    (args.output / 'passed.json').write_text(json.dumps(result, indent=2))


def limited_group():
    parent = cgroup_path()
    value = (parent / 'memory.max').read_text().strip()
    if value != 'max' and int(value) <= CAP_BYTES:
        return parent, False
    child = parent / f'hmm-{os.getpid()}'
    try:
        child.mkdir()
        if not (child / 'memory.max').exists():
            raise RuntimeError('memory controller is not delegated to children')
        (child / 'memory.max').write_text(str(CAP_BYTES))
        (child / 'memory.swap.max').write_text('0')
    except (OSError, RuntimeError) as error:
        if child.exists():
            child.rmdir()
        raise RuntimeError('Cannot enforce 100 GB: run in a container/job with cgroup memory.max '
                           '<= 100000000000, or delegate the memory controller. '
                           'No GPU model has been loaded.') from error
    return child, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/root/blockdata/models/qwen3-14b-fp8')
    parser.add_argument('--prompts', required=True, help='Existing L2 prepared prompts directory')
    parser.add_argument('--config', default='configs/text_l2.json')
    parser.add_argument('--output', type=Path, default=Path('runs/memsmoke'))
    parser.add_argument('--jobs', type=int, help='Default: production generate_chunk + 1')
    parser.add_argument('--spill-gb', type=float, default=90, help='Decimal GB; lower to exercise spilling')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--group', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.spill_gb < 97 or (args.jobs is not None and args.jobs < 1):
        parser.error('Need 0 < spill-gb < 97 and jobs >= 1')
    os.chdir(ROOT)
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    if not Path('/root/blockdata').is_dir():
        parser.error('/root/blockdata is not mounted')
    if args.output.exists():
        parser.error(f'Use a fresh output directory: {args.output}')
    group, created = limited_group()
    stop_bytes = min(97_000_000_000, int((group / 'memory.max').read_text()) * .97)
    args.output.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    if created:
        command += ['--group', str(group)]
    peak, process, stopped = 0, None, False
    before = (group / 'memory.events').read_text()
    try:
        print(f'cgroup={group}; memory.max={(group / "memory.max").read_text().strip()}; '
              f'spill<={args.spill_gb} GB; stop={stop_bytes / 1e9:g} GB', flush=True)
        with (args.output / 'run.log').open('w') as log, \
                (args.output / 'memory.jsonl').open('w', buffering=1) as samples:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                usage = memory_usage(group)
                peak = max(peak, usage)
                stat = os.statvfs('/dev/shm')
                samples.write(json.dumps({'time': time.time(), 'cgroup_bytes': usage,
                                          'shm_used_bytes': (stat.f_blocks-stat.f_bfree)*stat.f_frsize})+'\n')
                if usage >= stop_bytes:
                    # Cache monitor starts spilling at 90 GB. Stop only after that margin is exhausted.
                    stopped = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            code = process.wait()
        summary = {'returncode': code, 'peak_sampled_bytes': peak, 'stopped_at_guard': stopped,
                   'cgroup': str(group), 'memory_events_before': before,
                   'memory_events_after': (group / 'memory.events').read_text(),
                   'memory_peak': (group / 'memory.peak').read_text().strip()
                   if (group / 'memory.peak').exists() else None}
        (args.output / 'memory.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        print(f'Artifacts: {args.output}; generation log: {args.output / "run.log"}')
        if code or stopped:
            raise SystemExit(code if code > 0 else 1)
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        if created:
            try:
                group.rmdir()
            except OSError:
                print(f'cgroup retained until workers exit: {group}', file=sys.stderr)


if __name__ == '__main__':
    main()
