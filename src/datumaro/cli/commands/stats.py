# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.cli.util.errors import CliException, WrongRevpathError
from datumaro.components.errors import ConflictingCategoriesError, ProjectNotFoundError
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.util import dump_json_file, str_to_bool
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import generate_next_file_name, load_project, parse_full_revpath

__all__ = [
    "build_parser",
    "get_sensitive_args",
]


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Get project statistics",
        description="""
        Outputs various project statistics like image mean and std (RGB),
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
        help="Compute image mean and std (RGB) (default: %(default)s)",
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


def get_sensitive_args():
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

    try:
        dataset, target_project = parse_full_revpath(args.target, project)
    except WrongRevpathError as e:
        for p in e.problems:
            if isinstance(p, ConflictingCategoriesError):
                src_names = [src for src in project.working_tree.sources]
                raise CliException(
                    "There are more than two sources with heterogeneous categories in the project. "
                    "This prevents computing the statistics of the merged one. "
                    f"Please specify one of the sources in the project ({src_names}), "
                    f"such as `datum stats {src_names[0]}`"
                ) from e

        raise e

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
