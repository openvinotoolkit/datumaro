# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import os.path as osp

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.searcher import Searcher
from datumaro.components.visualizer import Visualizer
from datumaro.util.image import save_image
from datumaro.util.scope import scope_add

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Search similar data in dataset",
        description="""
        Applies data exploration to a dataset for image/text query.
        The command can be useful if you have to find similar data in dataset.
        |n
        In simple cases, when dataset query do not 
        |n
        
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument("target", nargs="?", default="project", help="Target dataset")
    parser.add_argument(
        "-q",
        "--query",
        dest="query",
        required=True,
        help="Image path or id of query to search similar data",
    )
    parser.add_argument("-topk", type=int, dest="topk", help="Number of similar results")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.add_argument(
        "-s", "--save", dest="save", default=True, help="Save searcher result as png"
    )

    parser.set_defaults(command=search_command)

    return parser


def get_sensitive_args():
    return {
        search_command: [
            "target",
            "query",
            "topk",
            "project_dir",
            "save",
        ]
    }


def search_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    project = scope_add(load_project(args.project_dir))

    dataset, _ = parse_full_revpath(args.target, project)

    searcher = Searcher(dataset)

    # Get query datasetitem through query path
    if osp.exists(args.query):
        query_datasetitem = dataset.get_datasetitem_by_path(args.query)
    else:
        query_datasetitem = args.query

    results = searcher.search_topk(query_datasetitem, args.topk)

    subset_list = []
    id_list = []
    result_path_list = []
    for result in results:
        subset_list.append(result.subset)
        id_list.append(result.id)
        result_path_list.append(result.media.path)
    print(f"Most similar {args.topk} results of query in dataset: {result_path_list}")

    visualizer = Visualizer(dataset, figsize=(20, 20), alpha=0)
    fig = visualizer.vis_gallery(id_list, subset_list)

    if args.save:
        import numpy as np

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        save_image(osp.join("./searcher.png"), data, create_dir=True)

    return 0
