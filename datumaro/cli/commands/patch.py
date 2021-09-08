# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.errors import ProjectNotFoundError

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Updates dataset from another one",
        description="""
        Updates items of the first dataset with items from the second one.
        By default, this command changes datasets in-place, but there is an
        option to specify output directory.|n
        This command can be applied to the current project targets or
        arbitrary datasets outside the project. Note that if the destination
        is read-only (e.g. if it is a project, stage or a cache entry),
        the output directory must be provided.|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <destination revpath> <patch revpath>|n
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
        Examples:|n
        - Update a VOC-like dataset with a COCO-like annotations:|n
        |s|s%(prog)s path/to/dataset1:voc path/to/dataset2:coco|n
        |n
        - Generate a patched dataset, based on a project:|n
        |s|s%(prog)s -o patched_proj1/ proj1/ proj2/|n
        |n
        - Update the "source1" the current project with a dataset:|n
        |s|s%(prog)s source1 path/to/dataset2:coco|n
        |n
        - Generate a patched source from a previous revision and a dataset:|n
        |s|s%(prog)s -o new_src2/ HEAD~2:source-2 path/to/dataset2:yolo
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('destination', help="Target dataset revpath")
    parser.add_argument('patch', help="Patch dataset revpath")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Output directory (default: save in-place)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory, "
            "if it is specified and not empty")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the 'current' project (default: current dir)")
    parser.set_defaults(command=patch_command)

    return parser

def patch_command(args):
    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
        dst_dir = osp.abspath(dst_dir)

    project = None
    try:
        project = load_project(args.project_dir)
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    destination_dataset = parse_full_revpath(args.destination, project)
    patch_dataset = parse_full_revpath(args.patch, project)

    destination_dataset.update(patch_dataset)

    # TODO: Probably, a better way to control export
    # options is needed. Maybe, just allow converter
    # parameters like in export.
    destination_dataset.save(save_dir=dst_dir,
        save_images=True) # avoid spontaneous removal of images

    log.info("Patched dataset has been saved to '%s'" % dst_dir)

    return 0
