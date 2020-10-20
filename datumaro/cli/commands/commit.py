# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('paths', nargs='*',
        help="Files to include in the commit (default: all tracked)")
    parser.add_argument('-m', '--message', required=True, help="Commit message")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=commit_command)

    return parser

def commit_command(args):
    project = load_project(args.project_dir)

    project.vcs.commit(args.paths, args.message)

    return 0
