
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from io import StringIO
import importlib
import os
import os.path as osp
import subprocess
import sys


def check_instruction_set(instruction):
    return instruction == str.strip(
        # Let's ignore a warning from bandit about using shell=True.
        # In this case it isn't a security issue and we use some
        # shell features like pipes.
        subprocess.check_output(
            'lscpu | grep -o "%s" | head -1' % instruction,
            shell=True).decode('utf-8') # nosec
    )

def import_foreign_module(name, path, package=None):
    module = None
    default_path = sys.path.copy()
    try:
        sys.path = [ osp.abspath(path), ] + default_path
        sys.modules.pop(name, None) # remove from cache
        module = importlib.import_module(name, package=package)
        sys.modules.pop(name) # remove from cache
    except Exception:
        raise
    finally:
        sys.path = default_path
    return module

@contextmanager
def suppress_output(stdout=True, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout:
            old_stdout = sys.stdout
            sys.stdout = devnull

        if stderr:
            old_stderr = sys.stderr
            sys.stderr = devnull

        try:
            yield
        finally:
            if stdout:
                sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr

@contextmanager
def catch_output():
    stdout = StringIO()
    stderr = StringIO()

    old_stdout = sys.stdout
    sys.stdout = stdout

    old_stderr = sys.stderr
    sys.stderr = stderr

    try:
        yield stdout, stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
