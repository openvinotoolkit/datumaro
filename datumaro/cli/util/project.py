
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os

from datumaro.components.project import Project
from datumaro.util import generate_next_name


def load_project(project_dir):
    return Project.load(project_dir)

def generate_next_file_name(basename, basedir='.', sep='.', ext=''):
    """
    If basedir does not contain basename, returns basename,
    otherwise generates a name by appending sep to the basename
    and the number, next to the last used number in the basedir for
    files with basename prefix. Optionally, appends ext.
    """

    return generate_next_name(os.listdir(basedir), basename, sep, ext)
