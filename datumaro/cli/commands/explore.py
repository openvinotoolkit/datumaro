# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
import shutil

import numpy as np

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.explorer import Explorer
from datumaro.components.visualizer import Visualizer
from datumaro.util.image import save_image
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Explore similar data of query in dataset",
        description="""
        Applies data exploration to a dataset for image/text query.
        The command can be useful if you have to find similar data in dataset.
        |n
        The current project (-p/--project) is used as a context for plugins
        and models. It is used when there is a dataset path in target.
        When not specified, the current project's working tree is used.|n
        |n
        Examples:|n
        - Explore top50 similar images of image query in COCO dataset:|n
        |s|s%(prog)s -q path/to/image.jpg -topk 50|n
        - Explore top50 similar images of text query, elephant, in COCO dataset:|n
        |s|s%(prog)s -q elephant -topk 50|n
        - Explore top50 similar images of image query list in COCO dataset:|n
        |s|s%(prog)s -q path/to/image1.jpg/ path/to/image2.jpg/ path/to/image3.jpg/ -topk 50|n
        - Explore top50 similar images of text query list in COCO dataset:|n
        |s|s%(prog)s -q motorcycle/ bus/ train/ -topk 50
        """,
        formatter_class=MultilineFormatter,
    )

    # parser.add_argument(
    #     "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    # )  # workaround for -- eaten by positionals
    parser.add_argument("target", nargs="?", help="Target dataset")
    parser.add_argument(
        "-q",
        "--query",
        dest="query",
        required=True,
        help="Image path or id of query to explore similar data",
    )
    parser.add_argument("-topk", type=int, dest="topk", help="Number of similar results")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", default=False, help="Save explorer result files on explore_result folder"
    )

    parser.set_defaults(command=explore_command)

    return parser


def get_sensitive_args():
    return {
        explore_command: [
            "target",
            "query",
            "topk",
            "project_dir",
        ]
    }


@scoped
def explore_command(args):
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
    for t in targets:
        target_dataset, _ = parse_full_revpath(t, project)
        source_datasets.append(target_dataset)
    dataset = project.load_hashkey(source_datasets)

    explorer = Explorer(dataset)
    project.save_hashkey(explorer._item_list)
    project.save()

    # Get query datasetitem through query path
    if osp.exists(args.query):
        query_datasetitem = dataset.get_datasetitem_by_path(args.query)
    else:
        query_datasetitem = args.query

    results = explorer.explore_topk(query_datasetitem, args.topk)

    result_path_list = []
    log.info(f"Most similar {args.topk} results of query in dataset")
    for result in results:
        path = getattr(result.media, "path", None)
        result_path_list.append(path)
        log.info(f"id: {result.id} | subset: {result.subset} | path : {path}")

    if args.save:
        saved_result_path = osp.join(args.project_dir, "explore_result")
        if osp.exists(saved_result_path):
            os.rmdir(saved_result_path)
        os.makedirs(saved_result_path)
        for result in results:
            if not osp.exists(osp.join(saved_result_path, result.subset)):
                os.makedirs(osp.join(saved_result_path, result.subset))
            shutil.copyfile(path, osp.join(saved_result_path, result.subset, result.id + '.jpg'))

    return 0
