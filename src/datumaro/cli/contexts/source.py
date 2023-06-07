# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

from datumaro.util.scope import scope_add, scoped

from ..commands.require_project.modification import add, import_, remove
from ..util import MultilineFormatter, add_subparser
from ..util.project import load_project


def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument("name", nargs="?", help="Source name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=info_command)

    return parser


def get_info_sensitive_args():
    return {
        info_command: [
            "name",
            "project_dir",
        ],
    }


@scoped
def info_command(args):
    project = scope_add(load_project(args.project_dir))

    if args.name:
        source = project.working_tree.sources[args.name]
        print(source)
    else:
        for name, conf in project.working_tree.sources.items():
            print(name)
            if args.verbose:
                print(conf)


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        description="""
            Manipulate data sources inside of a project.|n
            |n
            A data source is a source of data for a project.
            The project combines multiple data sources into one dataset.
            The role of a data source is to provide dataset items - images
            and/or annotations.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter,
    )

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, "import", import_.build_parser)
    add_subparser(subparsers, "add", add.build_parser)
    add_subparser(subparsers, "remove", remove.build_parser)
    # TODO: we should distinguish this with datumaro/cli/project_commands/dataset_operations/info.py
    add_subparser(subparsers, "info", build_info_parser)

    return parser


def get_sensitive_args():
    return {
        **add.get_sensitive_args(),
        **import_.get_sensitive_args(),
        **remove.get_sensitive_args(),
        **get_info_sensitive_args(),
    }
