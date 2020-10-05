
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=unused-import

import argparse
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.project import \
    PROJECT_DEFAULT_CONFIG as DEFAULT_CONFIG
from datumaro.components.project import Project

from ..util import CliException, MultilineFormatter


def build_create_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Create empty project",
        description="""
            Create a new empty project.|n
            |n
            Examples:|n
            - Create a project in the current directory:|n
            |s|screate -n myproject|n
            |n
            - Create a project in other directory:|n
            |s|screate -o path/I/like/
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', default='.', dest='dst_dir',
        help="Save directory for the new project (default: current dir")
    parser.add_argument('-n', '--name', default=None,
        help="Name of the new project (default: same as project dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.set_defaults(command=create_command)

    return parser

def create_command(args):
    project_dir = osp.abspath(args.dst_dir)

    project_env_dir = osp.join(project_dir, DEFAULT_CONFIG.env_dir)
    if osp.isdir(project_env_dir) and os.listdir(project_env_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % project_env_dir)
        else:
            shutil.rmtree(project_env_dir, ignore_errors=True)

    own_dataset_dir = osp.join(project_dir, DEFAULT_CONFIG.dataset_dir)
    if osp.isdir(own_dataset_dir) and os.listdir(own_dataset_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % own_dataset_dir)
        else:
            # NOTE: remove the dir to avoid using data from previous project
            shutil.rmtree(own_dataset_dir)

    project_name = args.name
    if project_name is None:
        project_name = osp.basename(project_dir)

    log.info("Creating project at '%s'" % project_dir)

    Project.generate(project_dir, {
        'project_name': project_name,
    })

    log.info("Project has been created at '%s'" % project_dir)

    return 0
