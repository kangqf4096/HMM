"""Linux cgroup v2 accounting, including tmpfs and all child processes."""
from contextlib import contextmanager
from pathlib import Path

CAP_BYTES = 100_000_000_000  # 100 GB, deliberately below 100 GiB


def cgroup_path():
    relative = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                    if line.startswith('0::'))
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, right = line.split(' - ', 1)
        if right.split()[0] != 'cgroup2':
            continue
        fields = left.split()
        root, mount = fields[3], Path(fields[4])
        suffix = relative[len(root):] if relative.startswith(root.rstrip('/') + '/') else ''
        candidate = mount / suffix.lstrip('/')
        if (candidate / 'memory.current').is_file():
            return candidate
    raise RuntimeError('Cannot locate cgroup v2 memory controller.')


def memory_usage(group=None):
    return int(((group or cgroup_path()) / 'memory.current').read_text())


@contextmanager
def cache_lock(path=None):
    if not path:
        yield
        return
    import fcntl
    with Path(path).open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
