# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-m', '--message', required=True, help="Commit message")
    parser.add_argument('--allow-empty', action='store_true',
        help="Allow commits with no changes (default: %(default)s)")
    parser.add_argument('--no-cache', action='store_true',
        help="Don't put committed datasets into cache, "
            "save only metainfo (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=commit_command)

    return parser

def commit_command(args):
    project = load_project(args.project_dir)

    old_tree = project.head

    new_commit = project.commit(args.message)

    new_tree = project.working_tree
    diff = project.diff(old_tree, new_tree)

    print("Moved to commit '%s' %s" % (new_commit, args.message))
    print(" %s targets changed" % len(diff))
    for t, s in diff.items():
        print(" %s %s" % (s.name, t))

    return 0
