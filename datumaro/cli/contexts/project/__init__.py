# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import os
import os.path as osp
import shutil
from enum import Enum

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_filter import DatasetItemEncoder
from datumaro.components.extractor import AnnotationType
from datumaro.components.operations import (DistanceComparator,
    ExactComparator, compute_ann_statistics, compute_image_statistics, mean_std)
from datumaro.components.project import \
    PROJECT_DEFAULT_CONFIG as DEFAULT_CONFIG
from datumaro.components.project import Environment, Project, BuildStageType
from datumaro.util import str_to_bool

from ...util import (CliException, MultilineFormatter, add_subparser,
    make_file_name)
from ...util.project import generate_next_file_name, load_project, generate_next_name
from .diff import DiffVisualizer


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

    parser.add_argument('target', default='project',
        help="Targets to do export for (default: '%(default)s')")
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
        converter = project.env.converters.get(args.format)
    except KeyError:
        raise CliException("Converter for format '%s' is not found" % \
            args.format)

    extra_args = converter.from_cmdline(args.extra_args)
    def converter_proxy(extractor, save_dir):
        return converter.convert(extractor, save_dir, **extra_args)

    if args.filter:
        filter_args = FilterModes.make_filter_args(args.filter_mode)
        filter_args['expr'] = args.filter

    log.info("Loading the project...")
    dataset = project.make_dataset(args.target)

    log.info("Exporting the project...")
    if args.filter:
        dataset = dataset.filter(**filter_args)
    dataset.export(converter_proxy, save_dir=dst_dir)

    log.info("Project exported to '%s'" % dst_dir)

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

    parser.add_argument('target', default='project', nargs='?',
        help="Project target to apply transform to (default: project)")
    parser.add_argument('-e', '--filter', default=None,
        help="XML XPath filter expression for dataset items")
    parser.add_argument('-m', '--mode', default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)" % \
            (', '.join(FilterModes.list_options()) , '%(default)s'))
    parser.add_argument('--dry-run', action='store_true',
        help="Print XML representations to be filtered and exit")
    parser.add_argument('--stage', type=str_to_bool, default=True,
        help="Include this action as a project build step (default: %(default)s)")
    parser.add_argument('--apply', type=str_to_bool, default=True,
        help="Run this action immediately (default: %(default)s)")
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

    filter_args = FilterModes.make_filter_args(args.mode)
    filter_args['expr'] = args.filter

    if args.dry_run:
        dataset = project.make_dataset(args.target)
        dataset = dataset.filter(**filter_args)
        for item in dataset:
            encoded_item = DatasetItemEncoder.encode(item, dataset.categories())
            xml_item = DatasetItemEncoder.to_string(encoded_item)
            print(xml_item)
        return 0

    if not args.filter:
        raise CliException("Expected a filter expression ('-e' argument)")

    project.build_targets.add_stage(args.target, {
        'type': BuildStageType.filter.name,
        'params': dict(filter_args),
    })

    if args.apply:
        log.info("Filtering...")

        dataset = project.make_dataset(args.target)
        dataset.save(save_dir)

        log.info("Results have been saved to '%s'" % dst_dir)

    if args.stage:
        project.save()

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

def build_apply_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Apply some operations to project",
        description="""
            Applies several operations to a dataset
            and produces a new dataset.
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('file',
        help="Path to a file with a list of transforms and other actions")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save output (default: current dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('--build', action='store_true',
        help="Consider this invocation a build step")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=apply_command)

    return parser

def apply_command(args):
    project = load_project(args.project_dir)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('%s-apply' % \
            project.config.project_name)
    dst_dir = osp.abspath(dst_dir)

    pipeline = project.build_targets.read_pipeline(args.file)
    graph, head = project.build_targets.apply_pipeline(pipeline)
    head_node = graph.nodes[head]
    if head_node['config']['type'] != BuildStageType.convert.name:
        dataset = head_node['dataset']
        dataset.save(dst_dir, save_images=args.build)
    else:
        raise NotImplementedError()

    log.info("Results have been saved to '%s'" % dst_dir)

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

    parser.add_argument('target', default='project', nargs='?',
        help="Project target to apply transform to (default: project)")
    parser.add_argument('-t', '--transform', required=True,
        help="Transform to apply to the project")
    parser.add_argument('-o', '--output-dir', dest='dst_dir', default=None,
        help="Directory to save output (default: current dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('--stage', type=str_to_bool, default=True,
        help="Include this action as a project build step (default: %(default)s)")
    parser.add_argument('--apply', type=str_to_bool, default=True,
        help="Run this action immediately (default: %(default)s)")
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
        transform = project.env.transforms.get(args.transform)
    except KeyError:
        raise CliException("Transform '%s' is not found" % args.transform)

    extra_args = {}
    if hasattr(transform, 'from_cmdline'):
        extra_args = transform.from_cmdline(args.extra_args)

    project.build_targets.add_stage(args.target, {
        'type': BuildStageType.transform.name,
        'kind': args.transform,
        'params': dict(extra_args),
    })

    if args.apply:
        log.info("Transforming...")

        dataset = project.make_dataset(args.target)
        dataset.save(dst_dir)

        log.info("Transform results have been saved to '%s'" % dst_dir)

    if args.stage:
        project.save()

    return 0

def build_build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Build project",
        description="""
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('target', default='project', nargs='?',
        help="Project target to apply transform to (default: project)")
    parser.add_argument('-f', '--force', action='store_true',
        help="Rerun build for the target, even if it has no changes")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=build_command)

    return parser

def build_command(args):
    project = load_project(args.project_dir)

    project.build(args.target, force=args.force)

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

    return parser
