# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
from enum import Enum, auto

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.operations import DistanceComparator, ExactComparator
from datumaro.util import dump_json_file
from datumaro.util.os_util import rmtree
from datumaro.util.scope import on_error_do, scope_add, scoped

from ..contexts.project.diff import DiffVisualizer
from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import generate_next_file_name, load_project, parse_full_revpath


class ComparisonMethod(Enum):
    equality = auto()
    distance = auto()


eq_default_if = ["id", "group"]  # avoid https://bugs.python.org/issue16399


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Compares two datasets",
        description="""
        Compares two datasets. This command has multiple forms:|n
        1) %(prog)s <revpath>|n
        2) %(prog)s <revpath> <revpath>|n
        |n
        1 - Compares the current project's main target ('project')
        in the working tree with the specified dataset.|n
        2 - Compares two specified datasets.|n
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
        Annotations can be matched 2 ways:|n
        - by equality checking|n
        - by distance computation|n
        |n
        Examples:|n
        - Compare two projects by distance, match boxes if IoU > 0.7,|n
        |s|s|s|ssave results to Tensorboard:|n
        |s|s%(prog)s other/project -o diff/ -f tensorboard --iou-thresh 0.7|n
        |n
        - Compare two projects for equality, exclude annotation groups |n
        |s|s|s|sand the 'is_crowd' attribute from comparison:|n
        |s|s%(prog)s other/project/ -if group -ia is_crowd|n
        |n
        - Compare two datasets, specify formats:|n
        |s|s%(prog)s path/to/dataset1:voc path/to/dataset2:coco|n
        |n
        - Compare the current working tree and a dataset:|n
        |s|s%(prog)s path/to/dataset2:coco|n
        |n
        - Compare a source from a previous revision and a dataset:|n
        |s|s%(prog)s HEAD~2:source-2 path/to/dataset2:yolo
        """,
        formatter_class=MultilineFormatter,
    )

    formats = ", ".join(f.name for f in DiffVisualizer.OutputFormat)
    comp_methods = ", ".join(m.name for m in ComparisonMethod)

    def _parse_output_format(s):
        try:
            return DiffVisualizer.OutputFormat[s.lower()]
        except KeyError:
            raise argparse.ArgumentError(
                "format",
                message="Unknown output " "format '%s', the only available are: %s" % (s, formats),
            )

    def _parse_comparison_method(s):
        try:
            return ComparisonMethod[s.lower()]
        except KeyError:
            raise argparse.ArgumentError(
                "method",
                message="Unknown comparison "
                "method '%s', the only available are: %s" % (s, comp_methods),
            )

    parser.add_argument("first_target", help="The first dataset revpath to be compared")
    parser.add_argument(
        "second_target", nargs="?", help="The second dataset revpath to be compared"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        default=None,
        help="Directory to save comparison results " "(default: generate automatically)",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=_parse_comparison_method,
        default=ComparisonMethod.equality.name,
        help="Comparison method, one of {} (default: %(default)s)".format(comp_methods),
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the current project (default: current dir)",
    )
    parser.set_defaults(command=diff_command)

    distance_parser = parser.add_argument_group("Distance comparison options")
    distance_parser.add_argument(
        "--iou-thresh",
        default=0.5,
        type=float,
        help="IoU match threshold for shapes (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=_parse_output_format,
        default=DiffVisualizer.DEFAULT_FORMAT.name,
        help="Output format, one of {} (default: %(default)s)".format(formats),
    )

    equality_parser = parser.add_argument_group("Equality comparison options")
    equality_parser.add_argument(
        "-iia", "--ignore-item-attr", action="append", help="Ignore item attribute (repeatable)"
    )
    equality_parser.add_argument(
        "-ia", "--ignore-attr", action="append", help="Ignore annotation attribute (repeatable)"
    )
    equality_parser.add_argument(
        "-if",
        "--ignore-field",
        action="append",
        help="Ignore annotation field (repeatable, default: %s)" % eq_default_if,
    )
    equality_parser.add_argument(
        "--match-images",
        action="store_true",
        help="Match dataset items by image pixels instead of ids",
    )
    equality_parser.add_argument("--all", action="store_true", help="Include matches in the output")

    return parser


def get_sensitive_args():
    return {
        diff_command: [
            "first_target",
            "second_target",
            "dst_dir",
            "project_dir",
        ],
    }


@scoped
def diff_command(args):
    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("diff")
    dst_dir = osp.abspath(dst_dir)

    if not osp.exists(dst_dir):
        on_error_do(rmtree, dst_dir, ignore_errors=True)
        os.makedirs(dst_dir)

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    try:
        if not args.second_target:
            first_dataset = project.working_tree.make_dataset()
            second_dataset, target_project = parse_full_revpath(args.first_target, project)
            if target_project:
                scope_add(target_project)
        else:
            first_dataset, target_project = parse_full_revpath(args.first_target, project)
            if target_project:
                scope_add(target_project)

            second_dataset, target_project = parse_full_revpath(args.second_target, project)
            if target_project:
                scope_add(target_project)
    except Exception as e:
        raise CliException(str(e))

    if args.method is ComparisonMethod.equality:
        if args.ignore_field:
            args.ignore_field = eq_default_if
        comparator = ExactComparator(
            match_images=args.match_images,
            ignored_fields=args.ignore_field,
            ignored_attrs=args.ignore_attr,
            ignored_item_attrs=args.ignore_item_attr,
        )
        matches, mismatches, a_extra, b_extra, errors = comparator.compare_datasets(
            first_dataset, second_dataset
        )

        output = {
            "mismatches": mismatches,
            "a_extra_items": sorted(a_extra),
            "b_extra_items": sorted(b_extra),
            "errors": errors,
        }
        if args.all:
            output["matches"] = matches

        output_file = osp.join(
            dst_dir, generate_next_file_name("diff", ext=".json", basedir=dst_dir)
        )
        log.info("Saving diff to '%s'" % output_file)
        dump_json_file(output_file, output, indent=True)

        print("Found:")
        print("The first project has %s unmatched items" % len(a_extra))
        print("The second project has %s unmatched items" % len(b_extra))
        print("%s item conflicts" % len(errors))
        print("%s matching annotations" % len(matches))
        print("%s mismatching annotations" % len(mismatches))
    elif args.method is ComparisonMethod.distance:
        comparator = DistanceComparator(iou_threshold=args.iou_thresh)

        with DiffVisualizer(
            save_dir=dst_dir, comparator=comparator, output_format=args.format
        ) as visualizer:
            log.info("Saving diff to '%s'" % dst_dir)
            visualizer.save(first_dataset, second_dataset)

    return 0
