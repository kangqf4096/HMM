"""Python 3.10 compatibility helpers.

The vLLM 0.8.4 image runs Python 3.10, but ``contextlib.chdir`` only exists
from Python 3.11.  Import ``chdir`` from here so the test suite runs on both.
"""
import os
from contextlib import contextmanager

try:  # Python >= 3.11
    from contextlib import chdir
except ImportError:  # Python 3.10

    @contextmanager
    def chdir(path):
        previous = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)
