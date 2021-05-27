# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import os
import os.path as osp
import shutil
import numpy as np
from enum import Enum

from datumaro.components.dataset_filter import DatasetItemEncoder
from datumaro.components.extractor import AnnotationType
from datumaro.components.operations import (DistanceComparator,
    ExactComparator, compute_ann_statistics, compute_image_statistics)
from datumaro.components.project import \
    PROJECT_DEFAULT_CONFIG as DEFAULT_CONFIG
from datumaro.components.project import Environment, Project
from datumaro.components.validator import validate_annotations, TaskType
from datumaro.util import error_rollback

from ...util import (CliException, MultilineFormatter, add_subparser,
    make_file_name)
from ...util.project import generate_next_file_name, load_project
from .diff import DatasetDiffVisualizer


def build_create_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Create empty project",
        description="""
            Create a new empty project.|n
            |n
            Examples:|n
            - Create a project in the current directory:|n
            |s|screate -n myproject|n
            |n
            - Create a project in other directory:|n
            |s|screate -o path/I/like/
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', default='.', dest='dst_dir',
        help="Save directory for the new project (default: current dir")
    parser.add_argument('-n', '--name', default=None,
        help="Name of the new project (default: same as project dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.set_defaults(command=create_command)

    return parser

def create_command(args):
    project_dir = osp.abspath(args.dst_dir)

    project_env_dir = osp.join(project_dir, DEFAULT_CONFIG.env_dir)
    if osp.isdir(project_env_dir) and os.listdir(project_env_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % project_env_dir)
        else:
            shutil.rmtree(project_env_dir, ignore_errors=True)

    own_dataset_dir = osp.join(project_dir, DEFAULT_CONFIG.dataset_dir)
    if osp.isdir(own_dataset_dir) and os.listdir(own_dataset_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % own_dataset_dir)
        else:
            # NOTE: remove the dir to avoid using data from previous project
            shutil.rmtree(own_dataset_dir)

    project_name = args.name
    if project_name is None:
        project_name = osp.basename(project_dir)

    log.info("Creating project at '%s'" % project_dir)

    Project.generate(project_dir, {
        'project_name': project_name,
    })

    log.info("Project has been created at '%s'" % project_dir)

    return 0

def build_import_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().importers.items)

    parser = parser_ctor(help="Create project from an existing dataset",
        description="""
            Creates a project from an existing dataset. The source can be:|n
            - a dataset in a supported format (check 'formats' section below)|n
            - a Datumaro project|n
            |n
            Formats:|n
            Datasets come in a wide variety of formats. Each dataset
            format defines its own data structure and rules on how to
            interpret the data. For example, the following data structure
            is used in COCO format:|n
            /dataset/|n
            - /images/<id>.jpg|n
            - /annotations/|n
            |n
            In Datumaro dataset formats are supported by
            Extractor-s and Importer-s.
            An Extractor produces a list of dataset items corresponding
            to the dataset. An Importer creates a project from the
            data source location.
            It is possible to add a custom Extractor and Importer.
            To do this, you need to put an Extractor and
            Importer implementation scripts to
            <project_dir>/.datumaro/extractors
            and <project_dir>/.datumaro/importers.|n
            |n
            List of builtin dataset formats: %s|n
            |n
            Examples:|n
            - Create a project from VOC dataset in the current directory:|n
            |s|simport -f voc -i path/to/voc|n
            |n
            - Create a project from COCO dataset in other directory:|n
            |s|simport -f coco -i path/to/coco -o path/I/like/
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', default='.', dest='dst_dir',
        help="Directory to save the new project to (default: current dir)")
    parser.add_argument('-n', '--name', default=None,
        help="Name of the new project (default: same as project dir)")
    parser.add_argument('--copy', action='store_true',
        help="Copy the dataset instead of saving source links")
    parser.add_argument('--skip-check', action='store_true',
        help="Skip source checking")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-i', '--input-path', required=True, dest='source',
        help="Path to import project from")
    parser.add_argument('-f', '--format',
        help="Source project format. Will try to detect, if not specified.")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for importer (pass '-- -h' for help)")
    parser.set_defaults(command=import_command)

    return parser

def import_command(args):
    project_dir = osp.abspath(args.dst_dir)

    project_env_dir = osp.join(project_dir, DEFAULT_CONFIG.env_dir)
    if osp.isdir(project_env_dir) and os.listdir(project_env_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % project_env_dir)
        else:
            shutil.rmtree(project_env_dir, ignore_errors=True)

    own_dataset_dir = osp.join(project_dir, DEFAULT_CONFIG.dataset_dir)
    if osp.isdir(own_dataset_dir) and os.listdir(own_dataset_dir):
        if not args.overwrite:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % own_dataset_dir)
        else:
            # NOTE: remove the dir to avoid using data from previous project
            shutil.rmtree(own_dataset_dir)

    project_name = args.name
    if project_name is None:
        project_name = osp.basename(project_dir)

    env = Environment()
    log.info("Importing project from '%s'" % args.source)

    extra_args = {}
    fmt = args.format
    if not args.format:
        if args.extra_args:
            raise CliException("Extra args can not be used without format")

        log.info("Trying to detect dataset format...")

        matches = env.detect_dataset(args.source)
        if len(matches) == 0:
            log.error("Failed to detect dataset format. "
                "Try to specify format with '-f/--format' parameter.")
            return 1
        elif len(matches) != 1:
            log.error("Multiple formats match the dataset: %s. "
                "Try to specify format with '-f/--format' parameter.",
                ', '.join(matches))
            return 1

        fmt = matches[0]
    elif args.extra_args:
        if fmt in env.importers:
            arg_parser = env.importers[fmt]
        elif fmt in env.extractors:
            arg_parser = env.extractors[fmt]
        else:
            raise CliException("Unknown format '%s'. A format can be added"
                "by providing an Extractor and Importer plugins" % fmt)

        if hasattr(arg_parser, 'parse_cmdline'):
            extra_args = arg_parser.parse_cmdline(args.extra_args)
        else:
            raise CliException("Format '%s' does not accept "
                "extra parameters" % fmt)

    log.info("Importing project as '%s'" % fmt)

    project = Project.import_from(osp.abspath(args.source), fmt, **extra_args)
    project.config.project_name = project_name
    project.config.project_dir = project_dir

    if not args.skip_check or args.copy:
        log.info("Checking the dataset...")
        dataset = project.make_dataset()
    if args.copy:
        log.info("Cloning data...")
        dataset.save(merge=True, save_images=True)
    else:
        project.save()

    log.info("Project has been created at '%s'" % project_dir)

    return 0


class FilterModes(Enum):
    # primary
    items = 1
    annotations = 2
    items_annotations = 3

    # shortcuts
    i = 1
    a = 2
    i_a = 3
    a_i = 3
    annotations_items = 3

    @staticmethod
    def parse(s):
        s = s.lower()
        s = s.replace('+', '_')
        return FilterModes[s]

    @classmethod
    def make_filter_args(cls, mode):
        if mode == cls.items:
            return {}
        elif mode == cls.annotations:
            return {
                'filter_annotations': True
            }
        elif mode == cls.items_annotations:
            return {
                'filter_annotations': True,
                'remove_empty': True,
            }
        else:
            raise NotImplementedError()

    @classmethod
    def list_options(cls):
        return [m.name.replace('_', '+') for m in cls]

def build_export_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().converters.items)

    parser = parser_ctor(help="Export project",
        description="""
            Exports the project dataset in some format. Optionally, a filter
            can be passed, check 'filter' command description for more info.
            Each dataset format has its own options, which
            are passed after '--' separator (see examples), pass '-- -h'
            for more info. If not stated otherwise, by default
            only annotations are exported, to include images pass
            '--save-images' parameter.|n
            |n
            Formats:|n
            In Datumaro dataset formats are supported by Converter-s.
            A Converter produces a dataset of a specific format
            from dataset items. It is possible to add a custom Converter.
            To do this, you need to put a Converter
            definition script to <project_dir>/.datumaro/converters.|n
            |n
            List of builtin dataset formats: %s|n
            |n
            Examples:|n
            - Export project as a VOC-like dataset, include images:|n
            |s|sexport -f voc -- --save-images|n
            |n
            - Export project as a COCO-like dataset in other directory:|n
            |s|sexport -f coco -o path/I/like/
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter)

    parser.add_argument('-e', '--filter', default=None,
        help="Filter expression for dataset items")
    parser.add_argument('--filter-mode', default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)" % \
            (', '.join(FilterModes.list_options()) , '%(default)s'))
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save output (default: a subdir in the current one)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('-f', '--format', required=True,
        help="Output format")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, default=None,
        help="Additional arguments for converter (pass '-- -h' for help)")
    parser.set_defaults(command=export_command)

    return parser

def export_command(args):
    project = load_project(args.project_dir)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('%s-%s' % \
            (project.config.project_name, make_file_name(args.format)))
    dst_dir = osp.abspath(dst_dir)

    try:
        converter = project.env.converters[args.format]
    except KeyError:
        raise CliException("Converter for format '%s' is not found" % \
            args.format)
    extra_args = converter.parse_cmdline(args.extra_args)

    filter_args = FilterModes.make_filter_args(args.filter_mode)

    log.info("Loading the project...")
    dataset = project.make_dataset()

    log.info("Exporting the project...")

    if args.filter:
        dataset = dataset.filter(args.filter, **filter_args)
    converter = project.env.converters[args.format]
    converter.convert(dataset, save_dir=dst_dir, **extra_args)

    log.info("Project exported to '%s' as '%s'" % (dst_dir, args.format))

    return 0

def build_filter_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Extract subproject",
        description="""
            Extracts a subproject that contains only items matching filter.
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
            Examples:|n
            - Filter images with width < height:|n
            |s|sextract -e '/item[image/width < image/height]'|n
            |n
            - Filter images with large-area bboxes:|n
            |s|sextract -e '/item[annotation/type="bbox" and
                annotation/area>2000]'|n
            |n
            - Filter out all irrelevant annotations from items:|n
            |s|sextract -m a -e '/item/annotation[label = "person"]'|n
            |n
            - Filter out all irrelevant annotations from items:|n
            |s|sextract -m a -e '/item/annotation[label="cat" and
            area > 99.5]'|n
            |n
            - Filter occluded annotations and items, if no annotations left:|n
            |s|sextract -m i+a -e '/item/annotation[occluded="True"]'
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-e', '--filter', default=None,
        help="XML XPath filter expression for dataset items")
    parser.add_argument('-m', '--mode', default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)" % \
            (', '.join(FilterModes.list_options()) , '%(default)s'))
    parser.add_argument('--dry-run', action='store_true',
        help="Print XML representations to be filtered and exit")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Output directory (default: update current project)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=filter_command)

    return parser

def filter_command(args):
    project = load_project(args.project_dir)

    if not args.dry_run:
        dst_dir = args.dst_dir
        if dst_dir:
            if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException("Directory '%s' already exists "
                    "(pass --overwrite to overwrite)" % dst_dir)
        else:
            dst_dir = generate_next_file_name('%s-filter' % \
                project.config.project_name)
        dst_dir = osp.abspath(dst_dir)

    dataset = project.make_dataset()

    filter_args = FilterModes.make_filter_args(args.mode)

    if args.dry_run:
        dataset = dataset.filter(expr=args.filter, **filter_args)
        for item in dataset:
            encoded_item = DatasetItemEncoder.encode(item, dataset.categories())
            xml_item = DatasetItemEncoder.to_string(encoded_item)
            print(xml_item)
        return 0

    if not args.filter:
        raise CliException("Expected a filter expression ('-e' argument)")

    dataset.filter_project(save_dir=dst_dir,
        filter_expr=args.filter, **filter_args)

    log.info("Subproject has been extracted to '%s'" % dst_dir)

    return 0

def build_merge_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Merge two projects",
        description="""
            Updates items of the current project with items
            from other project.|n
            |n
            Examples:|n
            - Update a project with items from other project:|n
            |s|smerge -p path/to/first/project path/to/other/project
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('other_project_dir',
        help="Path to a project")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Output directory (default: current project's dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=merge_command)

    return parser

def merge_command(args):
    first_project = load_project(args.project_dir)
    second_project = load_project(args.other_project_dir)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)

    first_dataset = first_project.make_dataset()
    second_dataset = second_project.make_dataset()

    first_dataset.update(second_dataset)
    first_dataset.save(save_dir=dst_dir)

    if dst_dir is None:
        dst_dir = first_project.config.project_dir
    dst_dir = osp.abspath(dst_dir)
    log.info("Merge results have been saved to '%s'" % dst_dir)

    return 0

def build_diff_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Compare projects",
        description="""
        Compares two projects, match annotations by distance.|n
        |n
        Examples:|n
        - Compare two projects, match boxes if IoU > 0.7,|n
        |s|s|s|sprint results to Tensorboard:
        |s|sdiff path/to/other/project -o diff/ -v tensorboard --iou-thresh 0.7
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('other_project_dir',
        help="Directory of the second project to be compared")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save comparison results (default: do not save)")
    parser.add_argument('-v', '--visualizer',
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
        dst_dir = generate_next_file_name('%s-%s-diff' % (
            first_project.config.project_name,
            second_project.config.project_name)
        )
    dst_dir = osp.abspath(dst_dir)
    log.info("Saving diff to '%s'" % dst_dir)

    if not osp.exists(dst_dir):
        on_error.do(shutil.rmtree, dst_dir, ignore_errors=True)

    with DatasetDiffVisualizer(save_dir=dst_dir, comparator=comparator,
            output_format=args.visualizer) as visualizer:
        visualizer.save(
            first_project.make_dataset(),
            second_project.make_dataset())

    return 0

_ediff_default_if = ['id', 'group'] # avoid https://bugs.python.org/issue16399

def build_ediff_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Compare projects for equality",
        description="""
        Compares two projects for equality.|n
        |n
        Examples:|n
        - Compare two projects, exclude annotation group |n
        |s|s|sand the 'is_crowd' attribute from comparison:|n
        |s|sediff other/project/ -if group -ia is_crowd
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('other_project_dir',
        help="Directory of the second project to be compared")
    parser.add_argument('-iia', '--ignore-item-attr', action='append',
        help="Ignore item attribute (repeatable)")
    parser.add_argument('-ia', '--ignore-attr', action='append',
        help="Ignore annotation attribute (repeatable)")
    parser.add_argument('-if', '--ignore-field', action='append',
        help="Ignore annotation field (repeatable, default: %s)" % \
            _ediff_default_if)
    parser.add_argument('--match-images', action='store_true',
        help='Match dataset items by images instead of ids')
    parser.add_argument('--all', action='store_true',
        help="Include matches in the output")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the first project to be compared (default: current dir)")
    parser.set_defaults(command=ediff_command)

    return parser

def ediff_command(args):
    first_project = load_project(args.project_dir)
    second_project = load_project(args.other_project_dir)

    if args.ignore_field:
        args.ignore_field = _ediff_default_if
    comparator = ExactComparator(
        match_images=args.match_images,
        ignored_fields=args.ignore_field,
        ignored_attrs=args.ignore_attr,
        ignored_item_attrs=args.ignore_item_attr)
    matches, mismatches, a_extra, b_extra, errors = \
        comparator.compare_datasets(
            first_project.make_dataset(), second_project.make_dataset())
    output = {
        "mismatches": mismatches,
        "a_extra_items": sorted(a_extra),
        "b_extra_items": sorted(b_extra),
        "errors": errors,
    }
    if args.all:
        output["matches"] = matches

    output_file = generate_next_file_name('diff', ext='.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4, sort_keys=True)

    print("Found:")
    print("The first project has %s unmatched items" % len(a_extra))
    print("The second project has %s unmatched items" % len(b_extra))
    print("%s item conflicts" % len(errors))
    print("%s matching annotations" % len(matches))
    print("%s mismatching annotations" % len(mismatches))

    log.info("Output has been saved to '%s'" % output_file)

    return 0

def build_transform_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().transforms.items)

    parser = parser_ctor(help="Transform project",
        description="""
            Applies some operation to dataset items in the project
            and produces a new project.|n
            |n
            Builtin transforms: %s|n
            |n
            Examples:|n
            - Convert instance polygons to masks:|n
            |s|stransform -t polygons_to_masks
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter)

    parser.add_argument('-t', '--transform', required=True,
        help="Transform to apply to the project")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save output (default: current dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, default=None,
        help="Additional arguments for transformation (pass '-- -h' for help)")
    parser.set_defaults(command=transform_command)

    return parser

def transform_command(args):
    project = load_project(args.project_dir)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('%s-%s' % \
            (project.config.project_name, make_file_name(args.transform)))
    dst_dir = osp.abspath(dst_dir)

    try:
        transform = project.env.transforms[args.transform]
    except KeyError:
        raise CliException("Transform '%s' is not found" % args.transform)

    extra_args = {}
    if hasattr(transform, 'parse_cmdline'):
        extra_args = transform.parse_cmdline(args.extra_args)

    log.info("Loading the project...")
    dataset = project.make_dataset()

    log.info("Transforming the project...")
    dataset.transform_project(
        method=transform,
        save_dir=dst_dir,
        **extra_args
    )

    log.info("Transform results have been saved to '%s'" % dst_dir)

    return 0

def build_stats_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Get project statistics",
        description="""
            Outputs various project statistics like image mean and std,
            annotations count etc.
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=stats_command)

    return parser

def stats_command(args):
    project = load_project(args.project_dir)

    dataset = project.make_dataset()
    stats = {}
    stats.update(compute_image_statistics(dataset))
    stats.update(compute_ann_statistics(dataset))

    dst_file = generate_next_file_name('statistics', ext='.json')
    log.info("Writing project statistics to '%s'" % dst_file)
    with open(dst_file, 'w') as f:
        json.dump(stats, f, indent=4, sort_keys=True)

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Get project info",
        description="""
            Outputs project info.
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('--all', action='store_true',
        help="Print all information")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=info_command)

    return parser

def info_command(args):
    project = load_project(args.project_dir)
    config = project.config
    env = project.env
    dataset = project.make_dataset()

    print("Project:")
    print("  name:", config.project_name)
    print("  location:", config.project_dir)
    print("Plugins:")
    print("  importers:", ', '.join(env.importers.items))
    print("  extractors:", ', '.join(env.extractors.items))
    print("  converters:", ', '.join(env.converters.items))
    print("  launchers:", ', '.join(env.launchers.items))

    print("Sources:")
    for source_name, source in config.sources.items():
        print("  source '%s':" % source_name)
        print("    format:", source.format)
        print("    url:", source.url)
        print("    location:", project.local_source_dir(source_name))

    def print_extractor_info(extractor, indent=''):
        print("%slength:" % indent, len(extractor))

        categories = extractor.categories()
        print("%scategories:" % indent, ', '.join(c.name for c in categories))

        for cat_type, cat in categories.items():
            print("%s  %s:" % (indent, cat_type.name))
            if cat_type == AnnotationType.label:
                print("%s    count:" % indent, len(cat.items))

                count_threshold = 10
                if args.all:
                    count_threshold = len(cat.items)
                labels = ', '.join(c.name for c in cat.items[:count_threshold])
                if count_threshold < len(cat.items):
                    labels += " (and %s more)" % (
                        len(cat.items) - count_threshold)
                print("%s    labels:" % indent, labels)

    print("Dataset:")
    print_extractor_info(dataset, indent="  ")

    subsets = dataset.subsets()
    print("  subsets:", ', '.join(subsets))
    for subset_name in subsets:
        subset = dataset.get_subset(subset_name)
        print("    subset '%s':" % subset_name)
        print_extractor_info(subset, indent="      ")

    print("Models:")
    for model_name, model in config.models.items():
        print("  model '%s':" % model_name)
        print("    type:", model.launcher)

    return 0

def build_validate_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Validate project",
        description="""
            Validates project based on specified task type and stores
            results like statistics, reports and summary in JSON file.
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('task_type',
        choices=[task_type.name for task_type in TaskType],
        help="Task type for validation")
    parser.add_argument('-s', '--subset', dest='subset_name', default=None,
        help="Subset to validate (default: None)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to validate (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER, default=None,
        help="Optional arguments for validator (pass '-- -h' for help)")
    parser.set_defaults(command=validate_command)

    return parser

def validate_command(args):
    project = load_project(args.project_dir)
    task_type = args.task_type
    subset_name = args.subset_name
    dst_file_name = f'validation_results-{task_type}'

    dataset = project.make_dataset()
    if subset_name is not None:
        dataset = dataset.get_subset(subset_name)
        dst_file_name += f'-{subset_name}'

    extra_args = {}
    from datumaro.components.validator import _Validator
    extra_args = _Validator.parse_cmdline(args.extra_args)
    validation_results = validate_annotations(dataset, task_type, **extra_args)

    def numpy_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()

    def _make_serializable(d):
        for key, val in list(d.items()):
            # tuple key to str
            if isinstance(key, tuple):
                d[str(key)] = val
                d.pop(key)
            if isinstance(val, dict):
                _make_serializable(val)

    _make_serializable(validation_results)

    dst_file = generate_next_file_name(dst_file_name, ext='.json')
    log.info("Writing project validation results to '%s'" % dst_file)
    with open(dst_file, 'w') as f:
        json.dump(validation_results, f, indent=4, sort_keys=True,
                  default=numpy_encoder)

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(
        description="""
            Manipulate projects.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter)

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, 'create', build_create_parser)
    add_subparser(subparsers, 'import', build_import_parser)
    add_subparser(subparsers, 'export', build_export_parser)
    add_subparser(subparsers, 'filter', build_filter_parser)
    add_subparser(subparsers, 'merge', build_merge_parser)
    add_subparser(subparsers, 'diff', build_diff_parser)
    add_subparser(subparsers, 'ediff', build_ediff_parser)
    add_subparser(subparsers, 'transform', build_transform_parser)
    add_subparser(subparsers, 'info', build_info_parser)
    add_subparser(subparsers, 'stats', build_stats_parser)
    add_subparser(subparsers, 'validate', build_validate_parser)

    return parser
