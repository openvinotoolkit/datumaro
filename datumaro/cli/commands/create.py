# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.project import Project
from datumaro.util.os_util import rmtree

from ..util import MultilineFormatter
from ..util.errors import CliException


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Create empty project",
        description="""
        Create an empty Datumaro project. A project is required for the most of
        Datumaro functionality.

        Examples:
        - Create a project in the current directory:

            %(prog)s

        - Create a project in other directory:

            %(prog)s -o path/I/like/
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', default='.', dest='dst_dir',
        help="Save directory for the new project (default: current dir")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.set_defaults(command=create_command)

    return parser

def get_sensitive_args():
    return {
        create_command: ['dst_dir',],
    }

def create_command(args):
    project_dir = osp.abspath(args.dst_dir)

    existing_project_dir = Project.find_project_dir(project_dir)
    if existing_project_dir and os.listdir(existing_project_dir):
        if args.overwrite:
            rmtree(existing_project_dir)
        else:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % existing_project_dir)

    log.info("Creating project at '%s'" % project_dir)

    Project.init(project_dir)

    log.info("Project has been created at '%s'" % project_dir)

    return 0
