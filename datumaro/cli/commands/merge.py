# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp
from collections import OrderedDict

from datumaro.components.dataset import DEFAULT_FORMAT
from datumaro.components.environment import Environment
from datumaro.components.errors import DatasetMergeError, DatasetQualityError, ProjectNotFoundError
from datumaro.components.operations import IntersectMerge
from datumaro.components.project import ProjectBuildTargets
from datumaro.util import dump_json_file
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter, join_cli_args
from ..util.errors import CliException
from ..util.project import generate_next_file_name, load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Merge few projects",
        description="""
        Merges multiple datasets into one and produces a new dataset.
        The command can be useful if you have few annotations and wish
        to merge them, taking into consideration potential overlaps and
        conflicts. This command can try to find common ground by voting or
        return a list of conflicts.|n
        |n
        In simple cases, when dataset images do not intersect and new
        labels are not added, the recommended way of merging is using the
        "patch" command. It will offer better performance and provide the same
        results.|n
        |n
        This command has multiple forms:|n
        1) %(prog)s <revpath>|n
        2) %(prog)s <revpath> <revpath> ...|n
        |n
        1 - Merges the current project's main target ('project')
        in the working tree with the specified dataset.|n
        2 - Merges the specified datasets.
        Note that the current project is not included in the list of merged
        sources automatically.|n
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
        The current project (-p/--project) is used as a context for plugins.
        It can be useful for dataset paths in targets. When not specified,
        the current project's working tree is used.|n
        |n
        The output format can be specified with the '-f/--format' option.
        Each dataset format has its own export
        options, which are passed after the '--' separator (see examples),
        pass '-- -h' for more info. If not stated otherwise, by default
        only annotations are exported; to include images pass
        '--save-images' parameter.|n
        |n
        Examples:|n
        - Merge annotations from 3 (or more) annotators:|n
        |s|s%(prog)s project1/ project2/ project3/|n
        |n
        - Check groups of the merged dataset for consistency:|n
        |s|s|slook for groups consising of 'person', 'hand' 'head', 'foot'|n
        |s|s%(prog)s project1/ project2/ -g 'person,hand?,head,foot?'|n
        |n
        - Merge two datasets, specify formats:|n
        |s|s%(prog)s path/to/dataset1:voc path/to/dataset2:coco|n
        |n
        - Merge the current working tree and a dataset:|n
        |s|s%(prog)s path/to/dataset2:coco|n
        |n
        - Merge a source from a previous revision and a dataset:|n
        |s|s%(prog)s HEAD~2:source-2 path/to/dataset2:yolo
        |n
        - Merge datasets and save in different format:|n
        |s|s%(prog)s -f voc dataset1/:yolo path2/:coco -- --save-images
        """,
        formatter_class=MultilineFormatter,
    )

    def _group(s):
        return s.split(",")

    parser.add_argument(
        "_positionals", nargs=argparse.REMAINDER, help=argparse.SUPPRESS
    )  # workaround for -- eaten by positionals
    parser.add_argument("targets", nargs="+", help="Target dataset revpaths (repeatable)")
    parser.add_argument(
        "-iou",
        "--iou-thresh",
        default=0.25,
        type=float,
        help="IoU match threshold for segments (default: %(default)s)",
    )
    parser.add_argument(
        "-oconf",
        "--output-conf-thresh",
        default=0.0,
        type=float,
        help="Confidence threshold for output " "annotations (default: %(default)s)",
    )
    parser.add_argument(
        "--quorum",
        default=0,
        type=int,
        help="Minimum count for a label and attribute voting "
        "results to be counted (default: %(default)s)",
    )
    parser.add_argument(
        "-g",
        "--groups",
        action="append",
        type=_group,
        help="A comma-separated list of labels in "
        "annotation groups to check. '?' postfix can be added to a label to "
        "make it optional in the group (repeatable)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        default=None,
        help="Output directory (default: generate a new one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-f", "--format", default=DEFAULT_FORMAT, help="Output format (default: %(default)s)"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the 'current' project (default: current dir)",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for exporter (pass '-- -h' for help). "
        "Must be specified after the main command arguments and after "
        "the '--' separator",
    )
    parser.set_defaults(command=merge_command)

    return parser


def get_sensitive_args():
    return {
        merge_command: ["targets", "project_dir", "dst_dir", "groups"],
    }


@scoped
def merge_command(args):
    # Workaround. Required positionals consume positionals from the end
    args._positionals += join_cli_args(args, "targets", "extra_args")

    has_sep = "--" in args._positionals
    if has_sep:
        pos = args._positionals.index("--")
        if pos == 0:
            raise argparse.ArgumentError(None, message="Expected at least 1 target argument")
    else:
        pos = len(args._positionals)
    args.targets = args._positionals[:pos] or [ProjectBuildTargets.MAIN_TARGET]
    args.extra_args = args._positionals[pos + has_sep :]

    show_plugin_help = "-h" in args.extra_args or "--help" in args.extra_args

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name("merged")
    dst_dir = osp.abspath(dst_dir)

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help and len(args.targets) == 1 and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        exporter = env.exporters[args.format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % args.format)

    export_args = exporter.parse_cmdline(args.extra_args)

    source_datasets = []
    try:
        if len(args.targets) == 1:
            source_datasets.append(project.working_tree.make_dataset())

        for t in args.targets:
            target_dataset, target_project = parse_full_revpath(t, project)
            if target_project:
                scope_add(target_project)
            source_datasets.append(target_dataset)
    except Exception as e:
        raise CliException(str(e))

    merger = IntersectMerge(
        conf=IntersectMerge.Conf(
            pairwise_dist=args.iou_thresh,
            groups=args.groups or [],
            output_conf_thresh=args.output_conf_thresh,
            quorum=args.quorum,
        )
    )
    merged_dataset = merger.merge(source_datasets)

    merged_dataset.export(save_dir=dst_dir, format=exporter, **export_args)

    report_path = osp.join(dst_dir, "merge_report.json")
    save_merge_report(merger, report_path)

    log.info("Merge results have been saved to '%s'" % dst_dir)
    log.info("Report has been saved to '%s'" % report_path)

    return 0


def save_merge_report(merger, path):
    item_errors = OrderedDict()
    source_errors = OrderedDict()
    all_errors = []

    for e in merger.errors:
        if isinstance(e, DatasetQualityError):
            item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1
        elif isinstance(e, DatasetMergeError):
            for s in e.sources:
                source_errors[str(s)] = source_errors.get(s, 0) + 1
            item_errors[str(e.item_id)] = item_errors.get(str(e.item_id), 0) + 1

        all_errors.append(str(e))

    errors = OrderedDict(
        [
            ("Item errors", item_errors),
            ("Source errors", source_errors),
            ("All errors", all_errors),
        ]
    )

    dump_json_file(path, errors, indent=True)
