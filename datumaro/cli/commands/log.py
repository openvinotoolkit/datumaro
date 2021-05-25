# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="Prints project history.")

    parser.add_argument('-n', '--count', default=10, type=int,
        help="Count of last commits to print (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=log_command)

    return parser

def log_command(args):
    project = load_project(args.project_dir)

    revisions = project.revs(args.count)
    if revisions:
        print('\n'.join('%s %s' % line for line in revisions))
    else:
        print("(Project history is empty)")

    return 0
