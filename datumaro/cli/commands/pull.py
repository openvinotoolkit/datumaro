# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-r', '--remote',
        help="Name of the remote")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=pull_command)

    return parser

def pull_command(args):
    project = load_project(args.project_dir)

    project.vcs.pull(remote=args.remote)

    return 0
