# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.environment import Environment
from datumaro.components.errors import ProjectNotFoundError
from datumaro.util.scope import scope_add, scoped

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Updates dataset from another one",
        description="""
        Updates items of the first dataset with items from the second one.
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating inplace, use the '--overwrite' parameter along with the
        '--save-images' export option (inplace updates fail by default
        to prevent data loss).|n
        |n
        Unlike the regular project data source joining, the datasets are not
        required to have the same labels. The labels from the "patch"
        dataset are projected onto the labels of the patched dataset,
        so only the annotations with the matching labels are used, i.e.
        all the annotations having unknown labels are ignored. Currently,
        this command doesn't allow to update the label information in the
        patched dataset.|n
        |n
        The command supports passing extra exporting options for the output
        dataset. The extra options should be passed after the main arguments
        and after the '--' separator. Particularly, this is useful to include
        images in the output dataset with '--save-images'.|n
        |n
        This command can be applied to the current project targets or
        arbitrary datasets outside a project. Note that if the destination
        is read-only (e.g. if it is a project, stage or a cache entry),
        the output directory must be provided.|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <target dataset revpath> <patch dataset revpath>|n
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
        Examples:|n
        - Update a VOC-like dataset with a COCO-like annotations:|n
        |s|s%(prog)s --overwrite dataset1/:voc dataset2/:coco -- --save-images|n
        |n
        - Generate a patched dataset, based on a project:|n
        |s|s%(prog)s -o patched_proj1/ proj1/ proj2/|n
        |n
        - Update the "source1" the current project with a dataset:|n
        |s|s%(prog)s -p proj/ --overwrite source1 path/to/dataset2:coco|n
        |n
        - Generate a patched source from a previous revision and a dataset:|n
        |s|s%(prog)s -o new_src2/ HEAD~2:source-2 path/to/dataset2:yolo|n
        |n
        - Update a dataset in a custom format, described in a project plugin:|n
        |s|s%(prog)s -p proj/ --overwrite dataset/:my_format dataset2/:coco
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('target', help="Target dataset revpath")
    parser.add_argument('patch', help="Patch dataset revpath")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Output directory (default: save in-place)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory, "
            "if it is not empty")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the 'current' project (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for exporting (pass '-- -h' for help). "
            "Must be specified after the main command arguments and after "
            "the '--' separator")
    parser.set_defaults(command=patch_command)

    return parser

@scoped
def patch_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    target_dataset, _project = parse_full_revpath(args.target, project)
    if _project is not None:
        scope_add(_project)

    try:
        converter = env.converters[target_dataset.format]
    except KeyError:
        raise CliException("Converter for format '%s' is not found" % \
            args.format)

    extra_args = converter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if not dst_dir:
        dst_dir = target_dataset.data_path
    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException("Directory '%s' already exists "
            "(pass --overwrite to overwrite)" % dst_dir)
    dst_dir = osp.abspath(dst_dir)

    patch_dataset, _project = parse_full_revpath(args.patch, project)
    if _project is not None:
        scope_add(_project)

    target_dataset.update(patch_dataset)

    target_dataset.save(save_dir=dst_dir, **extra_args)

    log.info("Patched dataset has been saved to '%s'" % dst_dir)

    return 0
