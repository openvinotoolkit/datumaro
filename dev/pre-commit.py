#!/usr/bin/env python3

# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

"""
This script is intended to be used as a pre-commit hook by contributors to this project.
It runs a subset of CI checks that is sufficiently fast to not be annoying when run
for every commit.

To install, make sure isort and black are installed on your system, then run:

$ ln -sv ../../dev/pre-commit.py ./.git/hooks/pre-commit
"""

import sys
import tempfile
from subprocess import call, check_call, check_output  # nosec B404


def main():
    success = True

    def try_call(args, **kwargs):
        nonlocal success

        if call(args, **kwargs) != 0:  # nosec B603
            success = False

    try_call(["git", "diff-index", "--check", "--cached", "HEAD"])

    diff_index_output = check_output(  # nosec B603, B607
        ["git", "diff-index", "-z", "--name-only", "--diff-filter=AM", "--cached", "HEAD"]
    )

    changed_files = []
    if diff_index_output:
        changed_files = diff_index_output.decode("utf-8").rstrip("\0").split("\0")

    changed_python_files = [f for f in changed_files if f.endswith(".py")]

    if changed_python_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            check_call(["git", "checkout-index", "-a", f"--prefix={temp_dir}/"])  # nosec B603, B607

            try_call(["isort", "--check", "--", *changed_python_files], cwd=temp_dir)
            try_call(["black", "--check", "--", *changed_python_files], cwd=temp_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
