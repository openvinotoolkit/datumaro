# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('targets', nargs='*',
        help="Names of sources, models, tracked files and dirs (default: all)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=push_command)

    return parser

def push_command(args):
    project = load_project(args.project_dir)

    project.vcs.push(targets=args.targets)

    return 0
