# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from ..util import MultilineFormatter, add_subparser
from ..util.project import load_project, generate_next_name


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Add a repository link")

    parser.add_argument('url', help="Repository url")
    parser.add_argument('-n', '--name', help="Name of the new remote")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")

    parser.set_defaults(command=add_command)

    return parser

def add_command(args):
    project = load_project(args.project_dir)

    name = args.name
    if not name:
        name = generate_next_name(project.vcs.repositories, 'remote',
            sep='-', default='1')
    project.vcs.repositories.add(name, args.url)
    project.save()

    log.info("Repository '%s' has been added to the project" % name)

    return 0

def build_remove_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Remove a repository link")

    parser.add_argument('name',
        help="Name of the repository to be removed")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=remove_command)

    return parser

def remove_command(args):
    project = load_project(args.project_dir)

    project.vcs.repositories.remove(args.name)
    project.save()

    return 0

def build_default_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Set or display the default repository")

    parser.add_argument('name', nargs='?',
        help="Name of the repository to set as default")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=default_command)

    return parser

def default_command(args):
    project = load_project(args.project_dir)

    if not args.name:
        default = project.vcs.repositories.get_default()
        if default:
            print(default)
        else:
            print("The default repository is not set.")

    else:
        project.vcs.repositories.set_default(args.name)
        project.save()

    return 0

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Display repository info")

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
        remote = project.vcs.repositories[args.name]
        print(remote)
    else:
        for name, conf in project.vcs.repositories.items():
            print(name)
            if args.verbose:
                print(conf)

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="""
            Manipulate repositories of the project.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter)

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, 'add', build_add_parser)
    add_subparser(subparsers, 'remove', build_remove_parser)
    add_subparser(subparsers, 'default', build_default_parser)
    add_subparser(subparsers, 'info', build_info_parser)

    return parser
