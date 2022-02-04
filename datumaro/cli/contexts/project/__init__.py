# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum
import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.dataset_filter import DatasetItemEncoder
from datumaro.components.environment import Environment
from datumaro.components.errors import MigrationError, ProjectNotFoundError
from datumaro.components.operations import (
    compute_ann_statistics, compute_image_statistics,
)
from datumaro.components.project import Project, ProjectBuildTargets
from datumaro.components.validator import TaskType
from datumaro.util import dump_json_file, str_to_bool
from datumaro.util.os_util import make_file_name
from datumaro.util.scope import scope_add, scoped

from ...util import MultilineFormatter, add_subparser
from ...util.errors import CliException
from ...util.project import (
    generate_next_file_name, load_project, parse_full_revpath,
)


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
    builtins = sorted(Environment().converters)

    parser = parser_ctor(help="Export project",
        description="""
        Exports a project in some format.|n
        |n
        Each dataset format has its own export
        options, which are passed after the '--' separator (see examples),
        pass '-- -h' for more info. If not stated otherwise, by default
        only annotations are exported, to include images pass
        '--save-images' parameter.|n
        |n
        A filter can be passed, check the 'filter' command description for
        more info.|n
        |n
        Formats:|n
        Datasets come in a wide variety of formats. Each dataset
        format defines its own data structure and rules on how to
        interpret the data. Check the user manual for the list of
        supported formats, examples and documentation.
        |n
        The list of supported formats can be extended by plugins.
        Check the "plugins" section of the developer guide for information
        about plugin implementation.|n
        |n
        List of builtin dataset formats: {}|n
        |n
        The command can only be applied to a project build target, a stage
        or the combined 'project' target, in which case all the targets will
        be affected.
        |n
        Examples:|n
        - Export project as a VOC-like dataset, include images:|n
        |s|s%(prog)s -f voc -- --save-images|n
        |n
        - Export project as a COCO-like dataset in other directory:|n
        |s|s%(prog)s -f coco -o path/I/like/
        """.format(', '.join(builtins)),
        formatter_class=MultilineFormatter)

    parser.add_argument('_positionals', nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS) # workaround for -- eaten by positionals
    parser.add_argument('target', nargs='?', default='project',
        help="A project target to be exported (default: %(default)s)")
    parser.add_argument('-e', '--filter',
        help="XML XPath filter expression for dataset items")
    parser.add_argument('--filter-mode', default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)" % \
            (', '.join(FilterModes.list_options()) , '%(default)s'))
    parser.add_argument('-o', '--output-dir', dest='dst_dir',
        help="Directory to save output (default: a subdir in the current one)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('-f', '--format', required=True,
        help="Output format")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for converter (pass '-- -h' for help). "
            "Must be specified after the main command arguments and after "
            "the '--' separator")
    parser.set_defaults(command=export_command)

    return parser

def get_export_sensitive_args():
    return {
        export_command: ['dst_dir', 'project_dir', 'name', 'extra_args',
            'target', 'filter'],
    }

@scoped
def export_command(args):
    has_sep = '--' in args._positionals
    if has_sep:
        pos = args._positionals.index('--')
        if 1 < pos:
            raise argparse.ArgumentError(None,
                message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or \
        [ProjectBuildTargets.MAIN_TARGET])[0]
    args.extra_args = args._positionals[pos + has_sep:]

    show_plugin_help = '-h' in args.extra_args or '--help' in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        converter = env.converters[args.format]
    except KeyError:
        raise CliException("Converter for format '%s' is not found" % \
            args.format)

    extra_args = converter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('export-%s' % \
            make_file_name(args.format))
    dst_dir = osp.abspath(dst_dir)

    if args.filter:
        filter_args = FilterModes.make_filter_args(args.filter_mode)
        filter_expr = args.filter

    log.info("Loading the project...")

    dataset = project.working_tree.make_dataset(args.target)
    if args.filter:
        dataset.filter(filter_expr, **filter_args)

    log.info("Exporting...")

    dataset.export(save_dir=dst_dir, format=converter, **extra_args)

    log.info("Results have been saved to '%s'" % dst_dir)

    return 0

def build_filter_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Extract subdataset",
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
        formatter_class=MultilineFormatter)

    parser.add_argument('target', nargs='?', default='project',
        help="Target dataset revpath (default: %(default)s)")
    parser.add_argument('-e', '--filter',
        help="XML XPath filter expression for dataset items")
    parser.add_argument('-m', '--mode', default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode (options: %s; default: %s)" % \
            (', '.join(FilterModes.list_options()) , '%(default)s'))
    parser.add_argument('--dry-run', action='store_true',
        help="Print XML representations to be filtered and exit")
    parser.add_argument('-o', '--output-dir', dest='dst_dir',
        help="""
            Output directory. Can be omitted for main project targets
            (i.e. data sources and the 'project' target, but not
            intermediate stages) and dataset targets.
            If not specified, the results will be saved inplace.
            """)
    parser.add_argument('--stage', type=str_to_bool, default=True,
        help="""
            Include this action as a project build step.
            If true, this operation will be saved in the project
            build tree, allowing to reproduce the resulting dataset later.
            Applicable only to main project targets (i.e. data sources
            and the 'project' target, but not intermediate stages)
            (default: %(default)s)
            """)
    parser.add_argument('--apply', type=str_to_bool, default=True,
        help="Run this command immediately. If disabled, only the "
            "build tree stage will be written (default: %(default)s)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=filter_command)

    return parser

def get_filter_sensitive_args():
    return {
        filter_command: ['target', 'filter', 'dst_dir', 'project_dir',],
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

    is_target = project is not None and \
        args.target in project.working_tree.build_targets
    if is_target:
        if not args.dst_dir and args.stage and (args.target != \
                ProjectBuildTargets.strip_target_name(args.target)):
            raise CliException("Adding a stage is only allowed for "
                "project targets, not their stages.")

        if args.target == ProjectBuildTargets.MAIN_TARGET:
            targets = list(project.working_tree.sources)
        else:
            targets = [args.target]

        build_tree = project.working_tree.clone()
        for target in targets:
            build_tree.build_targets.add_filter_stage(target,
                expr=filter_expr, params=filter_args)

    if args.apply:
        log.info("Filtering...")

        if is_target and not args.dst_dir:
            for target in targets:
                dataset = project.working_tree.make_dataset(
                    build_tree.make_pipeline(target))

                # Source might be missing in the working dir, so we specify
                # the output directory.
                # We specify save_images here as a heuristic. It can probably
                # be improved by checking if there are images in the dataset
                # directory.
                dataset.save(project.source_data_dir(target), save_images=True)

            log.info("Finished")
        else:
            dataset, _project = parse_full_revpath(args.target, project)
            if _project:
                scope_add(_project)

            dst_dir = args.dst_dir or dataset.data_path
            if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException("Directory '%s' already exists "
                    "(pass --overwrite to overwrite)" % dst_dir)
            dst_dir = osp.abspath(dst_dir)

            dataset.filter(filter_expr, *filter_args)
            dataset.save(dst_dir, save_images=True)

            log.info("Results have been saved to '%s'" % dst_dir)

    if is_target and args.stage:
        project.working_tree.config.update(build_tree.config)
        project.working_tree.save()

    return 0

def build_transform_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().transforms)

    parser = parser_ctor(help="Transform project",
        description="""
        Applies a batch operation to a dataset and produces a new dataset.|n
        |n
        By default, datasets are updated in-place. The '-o/--output-dir'
        option can be used to specify another output directory. When
        updating in-place, use the '--overwrite' parameter (in-place
        updates fail by default to prevent data loss), unless a project
        target is modified.|n
        |n
        Builtin transforms: {}|n
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
        - Convert instance polygons to masks:|n
        |s|s%(prog)s -t polygons_to_masks|n
        |n
        - Rename dataset items by a regular expression|n
        |s|s- Replace 'pattern' with 'replacement'|n|n
        |s|s%(prog)s -t rename -- -e '|pattern|replacement|'|n
        |n
        |s|s- Remove 'frame_' from item ids|n
        |s|s%(prog)s -t rename -- -e '|frame_(\\d+)|\\1|'|n
        |n
        - Split a dataset randomly:|n
        |s|s%(prog)s -t random_split --overwrite path/to/dataset:voc
        """.format(', '.join(builtins)),
        formatter_class=MultilineFormatter)

    parser.add_argument('_positionals', nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS) # workaround for -- eaten by positionals
    parser.add_argument('target', nargs='?', default='project',
        help="Target dataset revpath (default: project)")
    parser.add_argument('-t', '--transform', required=True,
        help="Transform to apply to the dataset")
    parser.add_argument('-o', '--output-dir', dest='dst_dir',
        help="""
            Output directory. Can be omitted for main project targets
            (i.e. data sources and the 'project' target, but not
            intermediate stages) and dataset targets.
            If not specified, the results will be saved inplace.
            """)
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('--stage', type=str_to_bool, default=True,
        help="""
            Include this action as a project build step.
            If true, this operation will be saved in the project
            build tree, allowing to reproduce the resulting dataset later.
            Applicable only to main project targets (i.e. data sources
            and the 'project' target, but not intermediate stages)
            (default: %(default)s)
            """)
    parser.add_argument('--apply', type=str_to_bool, default=True,
        help="Run this command immediately. If disabled, only the "
            "build tree stage will be written (default: %(default)s)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for transformation (pass '-- -h' for help). "
            "Must be specified after the main command arguments and after "
            "the '--' separator")
    parser.set_defaults(command=transform_command)

    return parser

def get_transform_sensitive_args():
    return {
        transform_command: ['dst_dir', 'project_dir', 'extra_args', 'target'],
    }

@scoped
def transform_command(args):
    has_sep = '--' in args._positionals
    if has_sep:
        pos = args._positionals.index('--')
        if 1 < pos:
            raise argparse.ArgumentError(None,
                message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or \
        [ProjectBuildTargets.MAIN_TARGET])[0]
    args.extra_args = args._positionals[pos + has_sep:]

    show_plugin_help = '-h' in args.extra_args or '--help' in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        transform = env.transforms[args.transform]
    except KeyError:
        raise CliException("Transform '%s' is not found" % args.transform)

    extra_args = transform.parse_cmdline(args.extra_args)

    is_target = project is not None and \
        args.target in project.working_tree.build_targets
    if is_target:
        if not args.dst_dir and args.stage and (args.target != \
                ProjectBuildTargets.strip_target_name(args.target)):
            raise CliException("Adding a stage is only allowed for "
                "project targets, not their stages.")

        if args.target == ProjectBuildTargets.MAIN_TARGET:
            targets = list(project.working_tree.sources)
        else:
            targets = [args.target]

        build_tree = project.working_tree.clone()
        for target in targets:
            build_tree.build_targets.add_transform_stage(target,
                args.transform, params=extra_args)

    if args.apply:
        log.info("Transforming...")

        if is_target and not args.dst_dir:
            for target in targets:
                dataset = project.working_tree.make_dataset(
                    build_tree.make_pipeline(target))

                # Source might be missing in the working dir, so we specify
                # the output directory
                # We specify save_images here as a heuristic. It can probably
                # be improved by checking if there are images in the dataset
                # directory.
                dataset.save(project.source_data_dir(target), save_images=True)

            log.info("Finished")
        else:
            dataset, _project = parse_full_revpath(args.target, project)
            if _project:
                scope_add(_project)

            dst_dir = args.dst_dir or dataset.data_path
            if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
                raise CliException("Directory '%s' already exists "
                    "(pass --overwrite to overwrite)" % dst_dir)
            dst_dir = osp.abspath(dst_dir)

            dataset.transform(args.transform, **extra_args)
            dataset.save(dst_dir, save_images=True)

            log.info("Results have been saved to '%s'" % dst_dir)

    if is_target and args.stage:
        project.working_tree.config.update(build_tree.config)
        project.working_tree.save()

    return 0

def build_stats_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Get project statistics",
        description="""
        Outputs various project statistics like image mean and std,
        annotations count etc.|n
        |n
        Target dataset is specified by a revpath. The full syntax is:|n
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
        - Compute project statistics:|n
        |s|s%(prog)s
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('target', default='project', nargs='?',
        help="Target dataset revpath (default: project)")
    parser.add_argument('-s', '--subset',
        help="Compute stats only for a specific subset")
    parser.add_argument('--image-stats', type=str_to_bool, default=True,
        help="Compute image mean and std (default: %(default)s)")
    parser.add_argument('--ann-stats', type=str_to_bool, default=True,
        help="Compute annotation statistics (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=stats_command)

    return parser

def get_stats_sensitive_args():
    return {
        stats_command: ['project_dir', 'target'],
    }

@scoped
def stats_command(args):
    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if args.project_dir:
            raise

    dataset, target_project = parse_full_revpath(args.target, project)
    if target_project:
        scope_add(target_project)

    if args.subset:
        dataset = dataset.get_subset(args.subset)

    stats = {}
    if args.image_stats:
        stats.update(compute_image_statistics(dataset))
    if args.ann_stats:
        stats.update(compute_ann_statistics(dataset))

    dst_file = generate_next_file_name('statistics', ext='.json')
    log.info("Writing project statistics to '%s'" % dst_file)
    dump_json_file(dst_file, stats, indent=True)

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Get project info",
        description="""
        Outputs project info - information about plugins,
        sources, build tree, models and revisions.|n
        |n
        Examples:|n
        - Print project info for the current working tree:|n
        |s|s%(prog)s|n
        |n
        - Print project info for the previous revision:|n
        |s|s%(prog)s HEAD~1
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('revision', default='', nargs='?',
        help="Target revision (default: current working tree)")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=info_command)

    return parser

def get_info_sensitive_args():
    return {
        info_command: ['project_dir', 'revision'],
    }

@scoped
def info_command(args):
    project = scope_add(load_project(args.project_dir))
    rev = project.get_rev(args.revision)
    env = rev.env

    print("Project:")
    print("  location:", project._root_dir)
    print("Plugins:")
    print("  extractors:", ', '.join(
        sorted(set(env.extractors) | set(env.importers))))
    print("  converters:", ', '.join(env.converters))
    print("  launchers:", ', '.join(env.launchers))

    print("Models:")
    for model_name, model in project.models.items():
        print("  model '%s':" % model_name)
        print("    type:", model.launcher)

    print("Sources:")
    for source_name, source in rev.sources.items():
        print("  '%s':" % source_name)
        print("    format:", source.format)
        print("    url:", osp.abspath(source.url) if source.url else '')
        print("    location:", osp.abspath(osp.join(
            project.source_data_dir(source_name), source.path)))
        print("    options:", source.options)

        print("    stages:")
        for stage in rev.build_targets[source_name].stages:
            print("      '%s':" % stage.name)
            print("        type:", stage.type)
            print("        hash:", stage.hash)
            print("        cached:",
                project.is_obj_cached(stage.hash) if stage.hash else 'n/a')
            if stage.kind:
                print("        kind:", stage.kind)
            if stage.params:
                print("        parameters:", stage.params)

    return 0

def build_validate_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Validate project",
        description="""
        Validates a dataset according to the task type and
        reports summary in a JSON file.|n
        Target dataset is specified by a revpath. The full syntax is:|n
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
        - Validate a project's subset as a classification dataset:|n
        |s|s%(prog)s -t classification -s train
        """,
        formatter_class=MultilineFormatter)

    task_types = ', '.join(t.name for t in TaskType)
    def _parse_task_type(s):
        try:
            return TaskType[s.lower()].name
        except:
            raise argparse.ArgumentTypeError("Unknown task type %s. Expected "
                "one of: %s" % (s, task_types))

    parser.add_argument('_positionals', nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS) # workaround for -- eaten by positionals
    parser.add_argument('target', default='project', nargs='?',
        help="Target dataset revpath (default: project)")
    parser.add_argument('-t', '--task',
        type=_parse_task_type, required=True,
        help="Task type for validation, one of %s" % task_types)
    parser.add_argument('-s', '--subset', dest='subset_name',
        help="Subset to validate (default: whole dataset)")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to validate (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Optional arguments for validator (pass '-- -h' for help)")
    parser.set_defaults(command=validate_command)

    return parser

def get_validate_sensitive_args():
    return {
        validate_command: ['target', 'project_dir', 'subset_name', 'extra_args'],
    }

@scoped
def validate_command(args):
    has_sep = '--' in args._positionals
    if has_sep:
        pos = args._positionals.index('--')
        if 1 < pos:
            raise argparse.ArgumentError(None,
                message="Expected no more than 1 target argument")
    else:
        pos = 1
    args.target = (args._positionals[:pos] or ['project'])[0]
    args.extra_args = args._positionals[pos + has_sep:]

    show_plugin_help = '-h' in args.extra_args or '--help' in args.extra_args

    project = None
    try:
        project = scope_add(load_project(args.project_dir))
    except ProjectNotFoundError:
        if not show_plugin_help and args.project_dir:
            raise

    if project is not None:
        env = project.env
    else:
        env = Environment()

    try:
        validator_type = env.validators[args.task]
    except KeyError:
        raise CliException("Validator type '%s' is not found" % args.task)

    extra_args = validator_type.parse_cmdline(args.extra_args)

    dataset, target_project = parse_full_revpath(args.target, project)
    if target_project:
        scope_add(target_project)

    dst_file_name = f'validation-report'
    if args.subset_name is not None:
        dataset = dataset.get_subset(args.subset_name)
        dst_file_name += f'-{args.subset_name}'

    validator = validator_type(**extra_args)
    report = validator.validate(dataset)

    def _make_serializable(d):
        for key, val in list(d.items()):
            # tuple key to str
            if isinstance(key, tuple):
                d[str(key)] = val
                d.pop(key)
            if isinstance(val, dict):
                _make_serializable(val)

    _make_serializable(report)

    dst_file = generate_next_file_name(dst_file_name, ext='.json')
    log.info("Writing project validation results to '%s'" % dst_file)
    dump_json_file(dst_file, report, indent=True, allow_numpy=True)

def build_migrate_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Migrate project",
        description="""
        Migrates the project from the old version to a new one.|n
        |n
        Examples:|n
        - Migrate a project from v1 to v2, save the new project in other dir:|n
        |s|s%(prog)s -o <output/dir>
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', dest='dst_dir', required=True,
        help="Output directory for the updated project")
    parser.add_argument('-f', '--force', action='store_true',
        help="Ignore source import errors (default: %(default)s)")
    parser.add_argument('-p', '--project', dest='project_dir',
        help="Directory of the project to migrate (default: current dir)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.set_defaults(command=migrate_command)

    return parser

def get_migrate_sensitive_args():
    return {
        migrate_command: ['dst_dir', 'project_dir',],
    }

@scoped
def migrate_command(args):
    dst_dir = args.dst_dir
    if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
        raise CliException("Directory '%s' already exists "
            "(pass --overwrite to overwrite)" % dst_dir)
    dst_dir = osp.abspath(dst_dir)

    log.debug("Migrating project from v1 to v2...")

    try:
        Project.migrate_from_v1_to_v2(args.project_dir, dst_dir,
            skip_import_errors=args.force)
    except Exception as e:
        raise MigrationError("Failed to migrate the project "
            "automatically. Try to create a new project and import sources "
            "manually with 'datum create' and 'datum import'.") from e

    log.debug("Finished")

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
    add_subparser(subparsers, 'export', build_export_parser)
    add_subparser(subparsers, 'filter', build_filter_parser)
    add_subparser(subparsers, 'transform', build_transform_parser)
    add_subparser(subparsers, 'info', build_info_parser)
    add_subparser(subparsers, 'stats', build_stats_parser)
    add_subparser(subparsers, 'validate', build_validate_parser)
    add_subparser(subparsers, 'migrate', build_migrate_parser)

    return parser

def get_sensitive_args():
    return {
        **get_export_sensitive_args(),
        **get_filter_sensitive_args(),
        **get_transform_sensitive_args(),
        **get_stats_sensitive_args(),
        **get_info_sensitive_args(),
        **get_validate_sensitive_args(),
        **get_migrate_sensitive_args(),
    }
