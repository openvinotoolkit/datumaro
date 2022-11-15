# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.dataset_filter import DatasetItemEncoder
from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import ProjectBuildTargets
from datumaro.util import str_to_bool
from datumaro.util.scope import scope_add, scoped

from ..contexts.project import FilterModes
from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import load_project, parse_full_revpath


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        help="Extract subdataset",
        description="""
        Extracts a subdataset that contains only items matching filter.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter (in-place
        updates fail by default to prevent data loss), unless a project
        target is modified.|n
        |n
        A filter is an XPath expression, which is applied to XML
        representation of a dataset item. Check '--dry-run' parameter
        to see XML representations of the dataset items.|n
        |n
        To filter annotations use the mode ('-m') parameter.|n
        Supported modes:|n
        - 'i', 'items'|n
        - 'a', 'annotations'|n
        - 'i+a', 'a+i', 'items+annotations', 'annotations+items'|n
        When filtering annotations, use the 'items+annotations'
        mode to point that annotation-less dataset items should be
        removed. To select an annotation, write an XPath that
        returns 'annotation' elements (see examples).|n
        |n
        This command has the following invocation syntax:
        - %(prog)s <target dataset revpath>|n
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
        The command can be applied to a dataset or a project build target,
        a stage or the combined 'project' target, in which case all the
        targets will be affected. A build tree stage will be recorded
        if '--stage' is enabled, and the resulting dataset(-s) will be
        saved if '--apply' is enabled.|n
        |n
        Examples:|n
        - Filter images with width < height:|n
        |s|s%(prog)s -e '/item[image/width < image/height]'|n
        |n
        - Filter images with large-area bboxes:|n
        |s|s%(prog)s -e '/item[annotation/type="bbox" and
            annotation/area>2000]'|n
        |n
        - Filter out all irrelevant annotations from items:|n
        |s|s%(prog)s -m a -e '/item/annotation[label = "person"]'|n
        |n
        - Filter out all irrelevant annotations from items:|n
        |s|s%(prog)s -m a -e '/item/annotation[label="cat" and
        area > 99.5]'|n
        |n
        - Filter occluded annotations and items, if no annotations left:|n
        |s|s%(prog)s -m i+a -e '/item/annotation[occluded="True"]'|n
        |n
        - Filter a VOC-like dataset inplace:|n
        |s|s%(prog)s -e '/item/annotation[label = "bus"]' --overwrite dataset/:voc
        """,
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "target", nargs="?", default="project", help="Target dataset revpath (default: %(default)s)"
    )
    parser.add_argument("-e", "--filter", help="XML XPath filter expression for dataset items")
    parser.add_argument(
        "-m",
        "--mode",
        default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)"
        % (", ".join(FilterModes.list_options()), "%(default)s"),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print XML representations to be filtered and exit"
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
    parser.add_argument(
        "--apply",
        type=str_to_bool,
        default=True,
        help="Run this command immediately. If disabled, only the "
        "build tree stage will be written (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="project_dir",
        help="Directory of the project to operate on (default: current dir)",
    )
    parser.set_defaults(command=filter_command)

    return parser


def get_sensitive_args():
    return {
        filter_command: [
            "target",
            "filter",
            "dst_dir",
            "project_dir",
        ],
    }


@scoped
def filter_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    filter_args = FilterModes.make_filter_args(args.mode)
    filter_expr = args.filter

    if args.dry_run:
        dataset, _project = parse_full_revpath(args.target, project)
        if _project:
            scope_add(_project)

        dataset = dataset.filter(expr=filter_expr, **filter_args)

        for item in dataset:
            encoded_item = DatasetItemEncoder.encode(item, dataset.categories())
            xml_item = DatasetItemEncoder.to_string(encoded_item)
            print(xml_item)
        return 0

    if not args.filter:
        raise CliException("Expected a filter expression ('-e' argument)")

    is_target = project is not None and args.target in project.working_tree.build_targets
    if is_target:
        if (
            not args.dst_dir
            and args.stage
            and (args.target != ProjectBuildTargets.strip_target_name(args.target))
        ):
            raise CliException(
                "Adding a stage is only allowed for " "project targets, not their stages."
            )

        if args.target == ProjectBuildTargets.MAIN_TARGET:
            targets = list(project.working_tree.sources)
        else:
            targets = [args.target]

        build_tree = project.working_tree.clone()
        for target in targets:
            build_tree.build_targets.add_filter_stage(target, expr=filter_expr, params=filter_args)

    if args.apply:
        log.info("Filtering...")

        if is_target and not args.dst_dir:
            for target in targets:
                dataset = project.working_tree.make_dataset(build_tree.make_pipeline(target))

                # Source might be missing in the working dir, so we specify
                # the output directory.
                # We specify save_media here as a heuristic. It can probably
                # be improved by checking if there are images in the dataset
                # directory.
                dataset.save(project.source_data_dir(target), save_media=True)

            log.info("Finished")
        else:
            dataset, _project = parse_full_revpath(args.target, project)
            if _project:
                scope_add(_project)

            dst_dir = args.dst_dir or dataset.data_path
            if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException(
                    "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
                )
            dst_dir = osp.abspath(dst_dir)

            dataset.filter(filter_expr, *filter_args)
            dataset.save(dst_dir, save_media=True)

            log.info("Results have been saved to '%s'" % dst_dir)

    if is_target and args.stage:
        project.working_tree.config.update(build_tree.config)
        project.working_tree.save()

    return 0
