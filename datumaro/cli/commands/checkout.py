# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util import MultilineFormatter
from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Navigate to a revision",
        description="""
        Command forms:|n
        1) %(prog)s <revision>|n
        2) %(prog)s [--] <source1> ...|n
        3) %(prog)s <revision> [--] <source1> <source2> ...|n
        |n
        1 - Restores a revision and all the sources in the working directory.|n
        2, 3 - Restores only specified sources from the specified revision.|n
        |s|sThe current revision is used, when not set.|n
        |s|s"--" is optinally used to separate source names and revisions.|n
        |n
        Examples:|n
        - Restore the previous revision:|n
        |s|s%(prog)s HEAD~1 |n |n
        - Restore the saved version of a source in the working tree|n
        |s|s%(prog)s -- source-1 |n |n
        - Restore a previous version of a source|n
        |s|s%(prog)s 33fbfbe my-source
        """, formatter_class=MultilineFormatter)

    parser.add_argument('_positionals', nargs='+',
        help=argparse.SUPPRESS) # workaround for -- eaten by positionals
    parser.add_argument('rev', nargs='?',
        help="Commit or tag (default: current)")
    parser.add_argument('sources', nargs='*',
        help="Sources to restore (default: all)")
    parser.add_argument('-f', '--force', action='store_true',
        help="Ignore unsaved changes (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=checkout_command)

    return parser

def checkout_command(args):
    has_sep = '--' in args._positionals
    if has_sep:
        pos = args._positionals.index('--')
    else:
        pos = 1
    args.rev = (args._positionals[:pos] or [None])[0]
    args.sources = args._positionals[pos + has_sep:]
    if has_sep and not args.sources:
        raise argparse.ArgumentError('sources', message="When '--' is used, "
            "at least 1 source name must be specified")

    project = load_project(args.project_dir)

    project.checkout(rev=args.rev, sources=args.sources, force=args.force)

    return 0
