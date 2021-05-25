# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('rev', nargs='?',
        help="Commit or tag (default: current)")
    parser.add_argument('-f', '--force', action='store_true',
        help="Ignore unsaved changes (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=checkout_command)

    return parser

def checkout_command(args):
    project = load_project(args.project_dir)

    project.checkout(rev=args.rev, force=args.force)

    return 0
