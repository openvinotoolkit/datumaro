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

from ...project_commands import get_project_commands
from ...project_commands.dataset_operations import export, stats
from ...util import MultilineFormatter, add_subparser, make_subcommands_help
from ...util.errors import CliException
from ...util.project import generate_next_file_name, load_project, parse_full_revpath


def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Get project info",
        description="""
        Outputs project info - information about plugins,
        sources, build tree, models and revisions.|n
        |n
        Examples:|n
        - Print project info for the current working tree:|n |n
        |s|s%(prog)s|n
        |n
        - Print project info for the previous revision:|n |n
        |s|s%(prog)s HEAD~1
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "revision", default="", nargs="?", help="Target revision (default: current working tree)"
    )
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
        info_command: ["project_dir", "revision"],
    }


@scoped
def info_command(args):
    project = scope_add(load_project(args.project_dir))
    rev = project.get_rev(args.revision)
    env = rev.env

    print("Project:")
    print("  location:", project._root_dir)
    print("Plugins:")
    print("  extractors:", ", ".join(sorted(set(env.extractors) | set(env.importers))))
    print("  exporters:", ", ".join(env.exporters))
    print("  launchers:", ", ".join(env.launchers))

    print("Models:")
    for model_name, model in project.models.items():
        print("  model '%s':" % model_name)
        print("    type:", model.launcher)

    print("Sources:")
    for source_name, source in rev.sources.items():
        print("  '%s':" % source_name)
        print("    format:", source.format)
        print("    url:", osp.abspath(source.url) if source.url else "")
        print(
            "    location:",
            osp.abspath(osp.join(project.source_data_dir(source_name), source.path)),
        )
        print("    options:", source.options)

        print("    stages:")
        for stage in rev.build_targets[source_name].stages:
            print("      '%s':" % stage.name)
            print("        type:", stage.type)
            print("        hash:", stage.hash)
            print("        cached:", project.is_obj_cached(stage.hash) if stage.hash else "n/a")
            if stage.kind:
                print("        kind:", stage.kind)
            if stage.params:
                print("        parameters:", stage.params)

    return 0


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
        **get_info_sensitive_args(),
        **stats.get_sensitive_args(),
        **export.get_sensitive_args(),
        **get_migrate_sensitive_args(),
    }
