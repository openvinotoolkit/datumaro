# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import ProjectBuildTargets
from datumaro.util import str_to_bool
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().transforms)

    parser = parser_ctor(
        help="Transform project",
        description="""
        Applies a batch operation to a dataset and produces a new dataset.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter (in-place
        updates fail by default to prevent data loss), unless a project
        target is modified.|n
        |n
        Builtin transforms: {}|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <target dataset revpath>|n
        |n
        <revpath> - either a dataset path or a revision path. The full
        syntax is:|n
        - Dataset paths:|n
        |s|s- <dataset path>[ :<format> ]|n
        - Revision paths:|n
        |s|s- <project path> [ @<rev> ] [ :<target> ]|n
        |s|s- <rev> [ :<target> ]|n
        |s|s- <target>|n
        |n
        The current project (-p/--project) is also used as a context for
        plugins, so it can be useful for dataset paths having custom formats.
        When not specified, the current project's working tree is used.|n
        |n
        The command can be applied to a dataset or a project build target,
        a stage or the combined 'project' target, in which case all the
        targets will be affected. A build tree stage will be recorded
        if '--stage' is enabled, and the resulting dataset(-s) will be
        saved if '--apply' is enabled.|n
        |n
        Examples:|n
        - Convert instance polygons to masks:|n |n
        |s|s%(prog)s -t polygons_to_masks|n
        |n
        - Rename dataset items by a regular expression:|n |n
        |s|s- Replace 'pattern' with 'replacement':|n |n
        |s|s|s|s%(prog)s -t rename -- -e '|pattern|replacement|'|n
        |n
        |s|s- Remove 'frame_' from item ids:|n |n
        |s|s|s|s%(prog)s -t rename -- -e '|frame_(\\d+)|\\1|'|n
        |n
        - Split a dataset randomly:|n |n
        |s|s%(prog)s -t random_split --overwrite path/to/dataset:voc
        """.format(
            ", ".join(builtins)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument(
        "target", nargs="?", default="project", help="Target dataset revpath (default: project)"
    )
    parser.add_argument(
        "-t", "--transform", required=True, help="Transform to apply to the dataset"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="""
            Output directory. Can be omitted for main project targets
            (i.e. data sources and the 'project' target, but not
            intermediate stages) and dataset targets.
            If not specified, the results will be saved inplace.
            """,
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
    parser.add_argument(
        "--stage",
        type=str_to_bool,
        default=True,
        help="""
            Include this action as a project build step.
            If true, this operation will be saved in the project
            build tree, allowing to reproduce the resulting dataset later.
            Applicable only to main project targets (i.e. data sources
            and the 'project' target, but not intermediate stages)
            (default: %(default)s)
            """,
    )
    parser.add_argument(
        "--apply",
        type=str_to_bool,
        default=True,
        help="Run this command immediately. If disabled, only the "
        "build tree stage will be written (default: %(default)s)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for transformation (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=transform_command)

    return parser


def get_sensitive_args():
    return {
        transform_command: ["dst_dir", "project_dir", "extra_args", "target"],
    }


@scoped
def transform_command(args):
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
        if not show_plugin_help and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        transform = env.transforms[args.transform]
    except KeyError:
        raise CliException("Transform '%s' is not found" % args.transform)

    extra_args = transform.parse_cmdline(args.extra_args)

    is_target = project is not None and args.target in project.working_tree.build_targets
    if is_target:
        if (
            not args.dst_dir
            and args.stage
            and (args.target != ProjectBuildTargets.strip_target_name(args.target))
        ):
            raise CliException(
                "Adding a stage is only allowed for " "project targets, not their stages."
            )

        if args.target == ProjectBuildTargets.MAIN_TARGET:
            targets = list(project.working_tree.sources)
        else:
            targets = [args.target]

        build_tree = project.working_tree.clone()
        for target in targets:
            build_tree.build_targets.add_transform_stage(target, args.transform, params=extra_args)

    if args.apply:
        log.info("Transforming...")

        if is_target and not args.dst_dir:
            for target in targets:
                dataset = project.working_tree.make_dataset(build_tree.make_pipeline(target))

                # Source might be missing in the working dir, so we specify
                # the output directory
                # We specify save_media here as a heuristic. It can probably
                # be improved by checking if there are images in the dataset
                # directory.
                dataset.save(project.source_data_dir(target), save_media=True)

            log.info("Finished")
        else:
            dataset, _project = parse_full_revpath(args.target, project)
            if _project:
                scope_add(_project)

            dst_dir = args.dst_dir or dataset.data_path
            if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException(
                    "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
                )
            dst_dir = osp.abspath(dst_dir)

            dataset.transform(args.transform, **extra_args)
            dataset.save(dst_dir, save_media=True)

            log.info("Results have been saved to '%s'" % dst_dir)

    if is_target and args.stage:
        project.working_tree.config.update(build_tree.config)
        project.working_tree.save()

    return 0
