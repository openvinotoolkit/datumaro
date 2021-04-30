# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=refs_command)

    return parser

def refs_command(args):
    project = load_project(args.project_dir)

    print('Branches:', ', '.join(project.vcs.refs))

    tags = project.vcs.tags
    if tags:
        print('Tags:', ', '.join(tags))

    return 0
