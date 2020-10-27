# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from ..util.project import load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="""
        Pulls updates from remotes and updates build stages
        """)

    parser.add_argument('sources', nargs='+',
        help="Names of sources to update")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=update_command)

    return parser

def update_command(args):
    project = load_project(args.project_dir)

    for source in args.sources:
        if source not in project.sources:
            raise KeyError("Unknown source '%s'" % source)

    for source in args.sources:
        project.sources.pull(source)
        project.build_targets.build(source)

    return 0
