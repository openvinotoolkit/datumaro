# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="Prints project history.")

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=status_command)

    return parser

def status_command(args):
    project = load_project(args.project_dir)

    statuses = project.status()

    if project.branch:
        print("On branch '%s', commit %s" % (project.branch, project.head_rev))
    else:
        print("HEAD is detached at commit %s" % project.head_rev)

    if statuses:
        print('\n'.join('%s\t%s' % (s.name, p) for p, s in statuses.items()))
    else:
        print("Working directory clean")

    return 0
