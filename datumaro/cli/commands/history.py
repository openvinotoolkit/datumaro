# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=history_command)

    return parser

def history_command(args):
    project = load_project(args.project_dir)

    print(''.join('%s %s' % line for line in project.revs()))

    return 0
