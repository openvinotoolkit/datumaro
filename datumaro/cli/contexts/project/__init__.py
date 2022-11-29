# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
from enum import Enum

from datumaro.components.filter import DatasetItemEncoder
from datumaro.components.environment import Environment
from datumaro.components.errors import MigrationError, ProjectNotFoundError
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.components.project import Project, ProjectBuildTargets
from datumaro.components.validator import TaskType
from datumaro.util import dump_json_file, str_to_bool
from datumaro.util.os_util import make_file_name
from datumaro.util.scope import scope_add, scoped

from ...util import MultilineFormatter, add_subparser
from ...util.errors import CliException
from ...util.project import generate_next_file_name, load_project, parse_full_revpath


class FilterModes(Enum):
    # primary
    items = 1
    annotations = 2
    items_annotations = 3

    # shortcuts
    i = 1
    a = 2
    i_a = 3
    a_i = 3
    annotations_items = 3

    @staticmethod
    def parse(s):
        s = s.lower()
        s = s.replace("+", "_")
        return FilterModes[s]

    @classmethod
    def make_filter_args(cls, mode):
        if mode == cls.items:
            return {}
        elif mode == cls.annotations:
            return {"filter_annotations": True}
        elif mode == cls.items_annotations:
            return {
                "filter_annotations": True,
                "remove_empty": True,
            }
        else:
            raise NotImplementedError()

    @classmethod
    def list_options(cls):
        return [m.name.replace("_", "+") for m in cls]


def build_export_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().exporters)

    parser = parser_ctor(
        help="Export project",
        description="""
        Exports a project in some format.|n
        |n
        Each dataset format has its own export
        options, which are passed after the '--' separator (see examples),
        pass '-- -h' for more info. If not stated otherwise, by default
        only annotations are exported, to include images pass
        '--save-images' parameter.|n
        |n
        A filter can be passed, check the 'filter' command description for
        more info.|n
        |n
        Formats:|n
        Datasets come in a wide variety of formats. Each dataset
        format defines its own data structure and rules on how to
        interpret the data. Check the user manual for the list of
        supported formats, examples and documentation.
        |n
        The list of supported formats can be extended by plugins.
        Check the "plugins" section of the developer guide for information
        about plugin implementation.|n
        |n
        List of builtin dataset formats: {}|n
        |n
        The command can only be applied to a project build target, a stage
        or the combined 'project' target, in which case all the targets will
        be affected.
        |n
        Examples:|n
        - Export project as a VOC-like dataset, include images:|n
        |s|s%(prog)s -f voc -- --save-images|n
        |n
        - Export project as a COCO-like dataset in other directory:|n
        |s|s%(prog)s -f coco -o path/I/like/
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument(
        "target",
        nargs="?",
        default="project",
        help="A project target to be exported (default: %(default)s)",
    )
    parser.add_argument("-e", "--filter", help="XML XPath filter expression for dataset items")
    parser.add_argument(
        "--filter-mode",
        default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)"
        % (", ".join(FilterModes.list_options()), "%(default)s"),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: a subdir in the current one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument("-f", "--format", required=True, help="Output format")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for exporter (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=export_command)

    return parser


def get_export_sensitive_args():
    return {
        export_command: ["dst_dir", "project_dir", "name", "extra_args", "target", "filter"],
    }


@scoped
def export_command(args):
    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
        if 1 < pos:
            raise argparse.ArgumentError(None, message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or [ProjectBuildTargets.MAIN_TARGET])[0]
    args.extra_args = args._positionals[pos + has_sep :]

    show_plugin_help = "-h" in args.extra_args or "--help" in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        exporter = env.exporters[args.format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % args.format)

    extra_args = exporter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("export-%s" % make_file_name(args.format))
    dst_dir = osp.abspath(dst_dir)

    if args.filter:
        filter_args = FilterModes.make_filter_args(args.filter_mode)
        filter_expr = args.filter

    log.info("Loading the project...")

    dataset = project.working_tree.make_dataset(args.target)
    if args.filter:
        dataset.filter(filter_expr, **filter_args)

    log.info("Exporting...")

    dataset.export(save_dir=dst_dir, format=exporter, **extra_args)

    log.info("Results have been saved to '%s'" % dst_dir)

    return 0


def build_stats_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Get project statistics",
        description="""
        Outputs various project statistics like image mean and std,
        annotations count etc.|n
        |n
        Target dataset is specified by a revpath. The full syntax is:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        - Revision paths:|n
        |s|s- <project path> [ @<rev> ] [ :<target> ]|n
        |s|s- <rev> [ :<target> ]|n
        |s|s- <target>|n
        |n
        Both forms use the -p/--project as a context for plugins. It can be
        useful for dataset paths in targets. When not specified, the current
        project's working tree is used.|n
        |n
        Examples:|n
        - Compute project statistics:|n
        |s|s%(prog)s
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target", default="project", nargs="?", help="Target dataset revpath (default: project)"
    )
    parser.add_argument("-s", "--subset", help="Compute stats only for a specific subset")
    parser.add_argument(
        "--image-stats",
        type=str_to_bool,
        default=True,
        help="Compute image mean and std (default: %(default)s)",
    )
    parser.add_argument(
        "--ann-stats",
        type=str_to_bool,
        default=True,
        help="Compute annotation statistics (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=stats_command)

    return parser


def get_stats_sensitive_args():
    return {
        stats_command: ["project_dir", "target"],
    }


@scoped
def stats_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    dataset, target_project = parse_full_revpath(args.target, project)
    if target_project:
        scope_add(target_project)

    if args.subset:
        dataset = dataset.get_subset(args.subset)

    stats = {}
    if args.image_stats:
        stats.update(compute_image_statistics(dataset))
    if args.ann_stats:
        stats.update(compute_ann_statistics(dataset))

    dst_file = generate_next_file_name("statistics", ext=".json")
    log.info("Writing project statistics to '%s'" % dst_file)
    dump_json_file(dst_file, stats, indent=True)


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

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, "info", build_info_parser)
    add_subparser(subparsers, "stats", build_stats_parser)
    add_subparser(subparsers, "export", build_export_parser)
    add_subparser(subparsers, "migrate", build_migrate_parser)

    return parser


def get_sensitive_args():
    return {
        **get_info_sensitive_args(),
        **get_stats_sensitive_args(),
        **get_export_sensitive_args(),
        **get_migrate_sensitive_args(),
    }
