# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
from enum import Enum

from datumaro.components.environment import Environment
from datumaro.components.errors import MigrationError, ProjectNotFoundError
from datumaro.components.filter import DatasetItemEncoder
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.components.project import Project, ProjectBuildTargets
from datumaro.components.validator import TaskType
from datumaro.util import dump_json_file, str_to_bool
from datumaro.util.os_util import make_file_name
from datumaro.util.scope import scope_add, scoped

from ...commands import get_project_commands
from ...util import MultilineFormatter, add_subparser, make_subcommands_help
from ...util.errors import CliException
from ...util.project import generate_next_file_name, load_project, parse_full_revpath


def build_migrate_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Migrate project",
        description="""
        Migrates the project from the old version to a new one.|n
        |n
        Examples:|n
        - Migrate a project from v1 to v2, save the new project in other dir:|n
        |s|s%(prog)s -o <output/dir>
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        required=True,
        help="Output directory for the updated project",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Ignore source import errors (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to migrate (default: current dir)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.set_defaults(command=migrate_command)

    return parser


def get_migrate_sensitive_args():
    return {
        migrate_command: [
            "dst_dir",
            "project_dir",
        ],
    }


@scoped
def migrate_command(args):
    dst_dir = args.dst_dir
    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException(
            "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
        )
    dst_dir = osp.abspath(dst_dir)

    log.debug("Migrating project from v1 to v2...")

    try:
        Project.migrate_from_v1_to_v2(args.project_dir, dst_dir, skip_import_errors=args.force)
    except Exception as e:
        raise MigrationError(
            "Failed to migrate the project "
            "automatically. Try to create a new project and import sources "
            "manually with 'datum create' and 'datum import'."
        ) from e

    log.debug("Finished")


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        description="""
            Manipulate projects.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter,
    )

    subcommands_desc = ""
    known_commands = get_project_commands()
    help_line_start = max((len(e[0]) for e in known_commands), default=0)
    help_line_start = max((2 + help_line_start) // 4 + 1, 6) * 4  # align to tabs
    if known_commands:
        if subcommands_desc:
            subcommands_desc += "\n"
        subcommands_desc += make_subcommands_help(known_commands, help_line_start)
    if subcommands_desc:
        subcommands_desc += (
            "\nRun '%s COMMAND --help' for more information on a command." % parser.prog
        )

    # TODO: revisit project related commands for better code structure
    subparsers = parser.add_subparsers(
        title=subcommands_desc, description="", help=argparse.SUPPRESS
    )

    commands = get_project_commands()
    for command_name, command, _ in commands:
        if command is not None:
            add_subparser(subparsers, command_name, command.build_parser)

    add_subparser(subparsers, "migrate", build_migrate_parser)

    return parser


def get_sensitive_args():
    return {
        **get_migrate_sensitive_args(),
    }
