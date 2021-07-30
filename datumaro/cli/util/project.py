# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional
import os
import re

from datumaro.cli.util.errors import WrongRevpathError
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import Project
from datumaro.util import escape, unescape
from datumaro.util.os_util import generate_next_name


def load_project(project_dir):
    return Project(project_dir)

def generate_next_file_name(basename, basedir='.', sep='.', ext=''):
    """
    If basedir does not contain basename, returns basename,
    otherwise generates a name by appending sep to the basename
    and the number, next to the last used number in the basedir for
    files with basename prefix. Optionally, appends ext.
    """

    return generate_next_name(os.listdir(basedir), basename, sep, ext)

_dataset_revpath_regex = None
_full_revpath_regex = None
def parse_full_revpath(s: str, ctx_project: Optional[Project]):
    if ctx_project:
        env = ctx_project.env
    else:
        env = Environment()

    def parse_dataset_pathspec(s: str):
        global _dataset_revpath_regex
        if not _dataset_revpath_regex:
            _dataset_revpath_regex = re.compile(
                "%s(:%s)?" % \
                (
                    r"(?P<dataset_path>[^:]+)",
                    r"(?P<format>.+)"
                )
            )

        match = re.fullmatch(_dataset_revpath_regex, s)
        if not match:
            raise ValueError("Failed to recognize dataset pathspec in '%s'" % s)
        match = { k: unescape(v, escapes=escapes) if v else v
            for k, v in match.groupdict().items() }

        path = match["dataset_path"]
        format = match["format"]
        return Dataset.import_from(path, format, env=env)

    def parse_revspec(s: str):
        global _full_revpath_regex
        if not _full_revpath_regex:
            _full_revpath_regex = re.compile(
                "(%(proj_path)s(@%(rev)s)?(:%(source)s)?)" % \
                {
                    'proj_path': r"(?P<proj_path>[^@:]+)",
                    'rev': r"(?P<rev>[^:]+)",
                    'source': r"(?P<source>.+)",
                }
            )

        match = re.fullmatch(_full_revpath_regex, s)
        if not match:
            raise ValueError("Failed to recognize revspec in '%s'" % s)
        match = { k: unescape(v, escapes=escapes) if v else v
            for k, v in match.groupdict().items() }

        proj_path = match["proj_path"]
        rev = match["rev"]
        source = match["source"]

        assert proj_path
        if rev:
            project = load_project(proj_path)

            # proj_path is either proj_path or rev or source name
        elif Project.find_project_dir(proj_path):
            project = load_project(proj_path)
        elif ctx_project:
            project = ctx_project
            if project.is_ref(proj_path):
                rev = proj_path
            elif not source:
                source = proj_path
        else:
            raise ProjectNotFoundError("Failed to find project at '%s'. " \
                "Specify project path with '-p/--project' or in the "
                "target pathspec." % proj_path)

        tree = project.get_rev(rev)
        return tree.make_dataset(source)


    # Escape colons used in absolute paths on Windows
    escapes = [(':\\', r'%%driveroot_bs%%')]
    escapes = [(':/', r'%%driveroot_s%%')]
    s = escape(s, escapes=escapes)

    errors = []
    try:
        return parse_dataset_pathspec(s)
    except Exception as e:
        errors.append(e)

    try:
        return parse_revspec(s)
    except Exception as e:
        errors.append(e)

    raise WrongRevpathError(problems=errors)

def parse_local_revpath(revpath: str):
    sep_pos = revpath.find(':')
    if -1 < sep_pos:
        rev = revpath[:sep_pos]
        target = revpath[sep_pos + 1:]
    else:
        rev = ''
        target = revpath

    return rev, target
