# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('_positionals', nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS) # args can't be resolved automatically
    parser.add_argument('rev', nargs='?',
        help="Commit or tag (default: current)")
    parser.add_argument('targets', nargs='*',
        help="Names of sources, models, tracked files and dirs (default: all)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=checkout_command)

    return parser

def checkout_command(args):
    try:
        pos = args._positionals.index('--')
        has_sep = True
    except ValueError:
        pos = 1
        has_sep = False
    args.rev = args._positionals[:pos] or []
    args.targets = args._positionals[pos + has_sep:]

    project = load_project(args.project_dir)

    project.vcs.checkout(rev=args.rev, targets=args.targets)

    return 0
