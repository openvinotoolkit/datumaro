# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from ..util import CliException, MultilineFormatter, add_subparser
from ..util.project import load_project, generate_next_name


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('url', help="Path to the remote")
    parser.add_argument('-n', '--name',
        help="Name of the new remote")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")

    parser.set_defaults(command=add_command)

    return parser

def add_command(args):
    project = load_project(args.project_dir)

    name = args.name
    if not name:
        name = generate_next_name(project.vcs.remotes, 'remote',
            sep='-', default=1)
    project.vcs.remotes.add(name, { 'url': args.url })
    project.save()

    log.info("Remote '%s' has been added to the project" % name)

    return 0

def build_remove_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Remove remote from project",
        description="Remove a remote from project.")

    parser.add_argument('names', nargs='+',
        help="Names of the remotes to be removed")
    parser.add_argument('-f', '--force', action='store_true',
        help="Ignore possible errors during removal")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=remove_command)

    return parser

def remove_command(args):
    project = load_project(args.project_dir)

    if not args.names:
        raise CliException("Expected remote name")

    for name in args.names:
        project.vcs.remotes.remove(name, force=args.force)
    project.save()

    return 0

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('name', nargs='?',
        help="Remote name")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Show details")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=info_command)

    return parser

def info_command(args):
    project = load_project(args.project_dir)

    if args.name:
        remote = project.vcs.remotes[args.name]
        print(remote)
    else:
        for name, conf in project.vcs.remotes.items():
            print(name)
            if args.verbose:
                print(conf)

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="""
            Manipulate remote data sources of a project.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter)

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, 'add', build_add_parser)
    add_subparser(subparsers, 'remove', build_remove_parser)
    add_subparser(subparsers, 'info', build_info_parser)

    return parser
