# Copyright (C) 2019-2021 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import argparse
from typing import Any, Dict, Type, cast

from datumaro.components.annotation import LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.errors import DatasetMergeError, MissingObjectError, ProjectNotFoundError
from datumaro.components.extractor import AnnotationType
from datumaro.components.media import Image, MediaElement, MultiframeImage, PointCloud, Video
from datumaro.util import dump_json
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Prints dataset overview",
        description="""
        Prints info about the dataset at <revpath>, or about the current
        project's combined dataset, if none is specified.|n
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
        Both forms use the -p/--project as a context for plugins. It can be
        useful for dataset paths in targets. When not specified, the current
        project's working tree is used.|n
        |n
        Examples:|n
        - Print dataset info for the current project's working tree:|n
        |s|s%(prog)s|n
        |n
        - Print dataset info for a path and a format name:|n
        |s|s%(prog)s path/to/dataset:voc|n
        |n
        - Print dataset info for a source from a past revision in JSON format:|n
        |s|s%(prog)s --json HEAD~2:source-2
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target", nargs="?", default="project", metavar="revpath", help="Target dataset revpath"
    )
    parser.add_argument("--json", action="store_true", help="Print output data in JSON format")
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the current project (default: current dir)",
    )
    parser.set_defaults(command=info_command)

    return parser


def get_sensitive_args():
    return {
        info_command: [
            "target",
            "project_dir",
        ],
    }


@scoped
def info_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    dataset = None
    dataset_problem = ""
    try:
        # TODO: avoid computing working tree hashes
        dataset, target_project = parse_full_revpath(args.target, project)
        if target_project:
            scope_add(target_project)
    except DatasetMergeError as e:
        dataset_problem = (
            "Can't merge project sources automatically: %s. "
            "The conflicting sources are: %s" % (e, ", ".join(e.sources))
        )
    except MissingObjectError as e:
        dataset_problem = str(e)

    def render_media_type(t: Type[MediaElement]) -> str:
        mapping = {
            Image: "image",
            Video: "video",
            PointCloud: "point_cloud",
            MultiframeImage: "multiframe_image",
        }
        for k, v in mapping.items():
            if issubclass(t, k):
                return v
        return "unknown"

    def render_dataset_info(dataset: Dataset) -> Dict[str, Any]:
        result = {}

        result["format"] = dataset.format or "unknown"
        result["media type"] = render_media_type(dataset.media_type())
        result["length"] = len(dataset)

        categories = dataset.categories()
        result["categories"] = {}

        if AnnotationType.label in categories:
            labels_info = cast(LabelCategories, categories[AnnotationType.label])
            result["categories"]["count"] = len(labels_info)
            result["categories"]["labels"] = [
                {
                    "id": idx,
                    "name": label.name,
                    "parent": label.parent,
                    "attributes": list(label.attributes),
                }
                for idx, label in enumerate(labels_info)
            ]
        else:
            result["categories"]["count"] = 0
            result["categories"]["labels"] = []

        result["subsets"] = []
        for subset_name, subset in dataset.subsets().items():
            result["subsets"].append({"name": subset_name, "length": len(subset)})

        return result

    DEFAULT_INDENT = 4 * " "
    LIST_COUNT_THRESHOLD = 10

    def print_dataset_info(data: Dict[str, Any], *, indent: str = ""):
        def _print_basic_type(key: str, data: Dict[str, Any], *, indent: str = indent):
            print(f"{indent}{key}:", data[key])

        if "format" in data:
            _print_basic_type("format", data)

        _print_basic_type("media type", data)
        _print_basic_type("length", data)

        print(indent + "categories:")
        count_threshold = LIST_COUNT_THRESHOLD
        labels = data["categories"]["labels"]
        labels_repr = ", ".join(label["name"] for label in labels[:count_threshold])
        if count_threshold < len(labels):
            labels_repr += " (and %s more)" % (len(labels) - count_threshold)
        print(DEFAULT_INDENT + indent + "labels:", labels_repr)

        subsets = data["subsets"]
        if subsets:
            print("subsets:")
        for subset_info in subsets:
            print(DEFAULT_INDENT + indent + subset_info["name"] + ":")
            _print_basic_type("length", subset_info, indent=2 * DEFAULT_INDENT + indent)

    if dataset is not None:
        output_data = render_dataset_info(dataset)
        if args.json:
            print(dump_json(output_data, indent=True, append_newline=True).decode())
        else:
            print_dataset_info(output_data)
    else:
        print("Dataset info is not available: ", dataset_problem)

    return 0
