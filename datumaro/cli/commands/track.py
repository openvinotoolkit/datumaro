# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('paths', nargs='+',
        help="Track files or directories")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=track_command)

    return parser

def track_command(args):
    project = load_project(args.project_dir)

    project.vcs.add(args.paths)

    return 0
