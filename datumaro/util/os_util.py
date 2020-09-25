
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import importlib
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
