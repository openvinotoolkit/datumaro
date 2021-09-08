# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from datumaro.cli.util import MultilineFormatter
from datumaro.util.scope import scope_add, scoped

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Prints project status.",
        description="""
        This command prints the summary of the project changes between
        the working tree of a project and its HEAD revision.
        """,
        formatter_class=MultilineFormatter
    )

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=status_command)

    return parser

@scoped
def status_command(args):
    project = scope_add(load_project(args.project_dir))

    statuses = project.status()

    if project.branch:
        print("On branch '%s', commit %s" % (project.branch, project.head_rev))
    else:
        print("HEAD is detached at commit %s" % project.head_rev)

    if statuses:
        for target, status in statuses.items():
            print('%s\t%s' % (status.name, target))
    else:
        print("Working directory clean")

    return 0
