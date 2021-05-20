# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.operations import DistanceComparator
from datumaro.util import error_rollback
from datumaro.util.os_util import rmtree

from ..util import CliException, MultilineFormatter
from ..util.project import generate_next_file_name, load_project
from ..contexts.project.diff import DatasetDiffVisualizer


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Compare projects",
        description="""
        Compares two projects, match annotations by distance.|n
        |n
        Examples:|n
        - Compare two projects, match boxes if IoU > 0.7,|n
        |s|s|s|sprint results to Tensorboard:
        |s|sdiff path/to/other/project -o diff/ -f tensorboard --iou-thresh 0.7
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('other_project_dir',
        help="The second project or revision to be compared")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save comparison results (default: do not save)")
    parser.add_argument('-f', '--format',
        default=DatasetDiffVisualizer.DEFAULT_FORMAT.name,
        choices=[f.name for f in DatasetDiffVisualizer.OutputFormat],
        help="Output format (default: %(default)s)")
    parser.add_argument('--iou-thresh', default=0.5, type=float,
        help="IoU match threshold for detections (default: %(default)s)")
    parser.add_argument('--conf-thresh', default=0.5, type=float,
        help="Confidence threshold for detections (default: %(default)s)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the first project to be compared (default: current dir)")
    parser.set_defaults(command=diff_command)

    return parser

@error_rollback('on_error', implicit=True)
def diff_command(args):
    first_project = load_project(args.project_dir)
    second_project = load_project(args.other_project_dir)

    comparator = DistanceComparator(iou_threshold=args.iou_thresh)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('diff')
    dst_dir = osp.abspath(dst_dir)
    log.info("Saving diff to '%s'" % dst_dir)

    if not osp.exists(dst_dir):
        on_error.do(rmtree, dst_dir, ignore_errors=True)

    with DatasetDiffVisualizer(save_dir=dst_dir, comparator=comparator,
            output_format=args.format) as visualizer:
        visualizer.save(
            first_project.working_tree.make_dataset(),
            second_project.working_tree.make_dataset())

    return 0
