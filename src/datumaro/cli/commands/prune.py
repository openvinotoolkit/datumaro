# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.prune import Prune
from datumaro.util import str_to_bool
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Prune dataset and make a representative subset",
        description="""
        Apply data pruning to a dataset.
        The command can be useful if you have to extract representative subset.
        |n
        The current project (-p/--project) is used as a context for plugins
        and models. It is used when there is a dataset path in target.
        When not specified, the current project's working tree is used.|n
        |n
        Examples:|n
        - Prune dataset with selecting random and ratio 80%:|n
        |s|s%(prog)s -m random -r 0.8|n
        - Prune dataset with clustering in image hash and ratio 50%:|n
        |s|s%(prog)s -m query_clust -h img -r 0.5|
        - Prune dataset based on entropy with clustering in image hash and ratio 50%:|n
        |s|s%(prog)s -m entropy -h img -r 0.5|
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("target", nargs="?", help="Target dataset")
    parser.add_argument("-m", "--method", dest="method", help="")
    parser.add_argument("-r", "--ratio", type=float, dest="ratio", help="")
    parser.add_argument("--hash-type", type=str, dest="hash_type", default="img", help="")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
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
    parser.set_defaults(command=prune_command)

    return parser


def get_sensitive_args():
    return {
        prune_command: [
            "target",
            "method",
            "ratio",
            "hash_type",
            "project_dir",
            "dst_dir",
        ]
    }


@scoped
def prune_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    if args.target:
        targets = [args.target]
    else:
        targets = list(project.working_tree.sources)

    source_datasets = []
    for target in targets:
        target_dataset, _ = parse_full_revpath(target, project)
        source_datasets.append(target_dataset)

    prune = Prune(*source_datasets, cluster_method=args.method)
    for dataset in source_datasets:
        dataset_dir = dataset.data_path
        dataset.save(dataset_dir, save_media=True, save_hashkey_meta=True)

    result, dist = prune.get_pruned(args.ratio)

    if dist:
        for id_, subset, center_idx, dist_ in dist.items():
            log.info(f"ID : {id_}, subset={subset} {dist_} away from {center_idx}th center")

    dst_dir = args.dst_dir or dataset.data_path
    result.save(dst_dir, save_media=True)

    log.info("Results have been saved to '%s'" % dst_dir)
