# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from contextlib import (
    ExitStack, contextmanager, redirect_stderr, redirect_stdout,
)
from io import StringIO
from typing import Iterable, Optional
import importlib
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import unicodedata

try:
    # Declare functions to remove files and directories.
    #
    # Use rmtree from GitPython to avoid the problem with removal of
    # readonly files on Windows, which Git uses extensively
    # It double checks if a file cannot be removed because of readonly flag
    from git.util import rmfile, rmtree  # pylint: disable=unused-import
    import git.util
    git.util.HIDE_WINDOWS_KNOWN_ERRORS = False

except ModuleNotFoundError:
    from os import remove as rmfile # pylint: disable=unused-import
    from shutil import rmtree as rmtree # pylint: disable=unused-import

from . import cast

DEFAULT_MAX_DEPTH = 10

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
    finally:
        sys.path = default_path
    return module

def walk(path, max_depth=None):
    if max_depth is None:
        max_depth = DEFAULT_MAX_DEPTH

    baselevel = path.count(osp.sep)
    for dirpath, dirnames, filenames in os.walk(path, topdown=True):
        curlevel = dirpath.count(osp.sep)
        if baselevel + max_depth <= curlevel:
            dirnames.clear() # topdown=True allows to modify the list

        yield dirpath, dirnames, filenames

def copytree(src, dst):
    # Serves as a replacement for shutil.copytree().
    #
    # Shutil works very slow pre 3.8
    # https://docs.python.org/3/library/shutil.html#platform-dependent-efficient-copy-operations
    # https://bugs.python.org/issue33671

    if sys.version_info >= (3, 8):
        shutil.copytree(src, dst)
        return

    assert src and dst
    src = osp.abspath(src)
    dst = osp.abspath(dst)

    if not osp.isdir(src):
        raise FileNotFoundError("Source directory '%s' doesn't exist" % src)

    if osp.isdir(dst):
        raise FileExistsError("Destination directory '%s' already exists" % dst)

    dst_basedir = osp.dirname(dst)
    if dst_basedir:
        os.makedirs(dst_basedir, exist_ok=True)

    try:
        if sys.platform == 'windows':
            # Ignore
            #   B603: subprocess_without_shell_equals_true
            #   B607: start_process_with_partial_path
            # In this case we control what is called and command arguments
            # PATH overriding is considered low risk
            subprocess.check_output(["xcopy", src, dst, # nosec
                    "/s", "/e", "/q", "/y", "/i"],
                stderr=subprocess.STDOUT, universal_newlines=True)
        elif sys.platform == 'linux':
            # As above
            subprocess.check_output(["cp", "-r", '--', src, dst], # nosec
                stderr=subprocess.STDOUT, universal_newlines=True)
        else:
            shutil.copytree(src, dst)
    except subprocess.CalledProcessError as e:
        raise Exception("Failed to copy data. The command '%s' "
            "has failed with the following output: '%s'" % (e.cmd, e.stdout)) \
            from e

@contextmanager
def suppress_output(stdout: bool = True, stderr: bool = False):
    with open(os.devnull, 'w') as devnull, ExitStack() as es:
        if stdout:
            es.enter_context(redirect_stdout(devnull))
        elif stderr:
            es.enter_context(redirect_stderr(devnull))

        yield

@contextmanager
def catch_output():
    stdout = StringIO()
    stderr = StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield stdout, stderr

def dir_items(path, ext, truncate_ext=False):
    items = []
    for f in os.listdir(path):
        ext_pos = f.rfind(ext)
        if ext_pos != -1:
            if truncate_ext:
                f = f[:ext_pos]
            items.append(f)
    return items

def split_path(path):
    path = osp.normpath(path)
    parts = []

    while True:
        path, part = osp.split(path)
        if part:
            parts.append(part)
        else:
            if path:
                parts.append(path)
            break
    parts.reverse()

    return parts

def is_subpath(path: str, base: str) -> bool:
    """
    Tests if a path is subpath of another path or the paths are equal.
    """

    base = osp.abspath(base)
    path = osp.abspath(path)
    return osp.join(path, '').startswith(osp.join(base, ''))

def make_file_name(s: str) -> str:
    # adapted from
    # https://docs.djangoproject.com/en/2.1/_modules/django/utils/text/#slugify
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    s = s.decode()
    s = re.sub(r'[^\w\s-]', '', s).strip().lower()
    s = re.sub(r'[-\s]+', '-', s)
    return s

def generate_next_name(names: Iterable[str], basename: str,
        sep: str = '.', suffix: str = '', default: Optional[str] = None) -> str:
    """
    Generates the "next" name by appending a next index to the occurrence
    of the basename with the highest index in the input collection.

    Returns: next string name

    Example:

    Inputs:
        name_abc
        name_base
        name_base1
        name_base5

    Basename: name_base

    Output: name_base6
    """

    pattern = re.compile(r'%s(?:%s(\d+))?%s' % \
        tuple(map(re.escape, [basename, sep, suffix])))
    matches = [match for match in (pattern.match(n) for n in names) if match]

    max_idx = max([cast(match[1], int, 0) for match in matches], default=None)
    if max_idx is None:
        if default is not None:
            idx = sep + str(default)
        else:
            idx = ''
    else:
        idx = sep + str(max_idx + 1)
    return basename + idx + suffix
