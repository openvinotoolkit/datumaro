# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple
import os
import re

from datumaro.cli.util.errors import WrongRevpathError
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import Project, Revision
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

def parse_dataset_pathspec(s: str,
        env: Optional[Environment] = None) -> Dataset:
    """
    Parses Dataset paths. The syntax is:
    - <dataset path>[ :<format> ]

    Returns: a dataset from the parsed path
    """

    match = re.fullmatch(r"""
        (?P<dataset_path>(?: [^:] | :[/\\] )+)
        (:(?P<format>.+))?
        """, s, flags=re.VERBOSE)
    if not match:
        raise ValueError("Failed to recognize dataset pathspec in '%s'" % s)
    match = match.groupdict()

    path = match["dataset_path"]
    format = match["format"]
    return Dataset.import_from(path, format, env=env)

def parse_revspec(s: str, ctx_project: Optional[Project] = None) -> Dataset:
    """
    Parses Revision paths. The syntax is:
    - <project path> [ @<rev> ] [ :<target> ]
    - <rev> [ :<target> ]
    - <target>
    The second and the third forms assume an existing "current" project.

    Returns: a dataset from the parsed path
    """

    match = re.fullmatch(r"""
        (?P<proj_path>(?: [^@:] | :[/\\] )+)
        (@(?P<rev>[^:]+))?
        (:(?P<source>.+))?
        """, s, flags=re.VERBOSE)
    if not match:
        raise ValueError("Failed to recognize revspec in '%s'" % s)
    match = match.groupdict()

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

def parse_full_revpath(s: str, ctx_project: Optional[Project]) -> Dataset:
    """
    revpath - either a Dataset path or a Revision path.

    Returns: a dataset from the parsed path
    """

    if ctx_project:
        env = ctx_project.env
    else:
        env = Environment()

    errors = []
    try:
        return parse_dataset_pathspec(s, env=env)
    except Exception as e:
        errors.append(e)

    try:
        return parse_revspec(s, ctx_project=ctx_project)
    except Exception as e:
        errors.append(e)

    raise WrongRevpathError(problems=errors)

def split_local_revpath(revpath: str) -> Tuple[Revision, str]:
    """
    Splits the given string into revpath components.

    A local revpath is a path to a revision withing the current project.
    The syntax is:
    - [ <revision> : ] [ <target> ]
    At least one part must be present.

    Returns: (revision, build target)
    """

    sep_pos = revpath.find(':')
    if -1 < sep_pos:
        rev = revpath[:sep_pos]
        target = revpath[sep_pos + 1:]
    else:
        rev = ''
        target = revpath

    return rev, target
