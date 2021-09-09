# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os

from datumaro.components.errors import ProjectNotFoundError
from datumaro.components.project import Environment
from datumaro.util.scope import on_error_do, scope_add, scoped

from ..util import MultilineFormatter, add_subparser, join_cli_args
from ..util.errors import CliException
from ..util.project import generate_next_name, load_project


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    env = Environment()
    builtins = sorted(set(env.extractors) | set(env.importers))

    parser = parser_ctor(help="Add data source to project",
        description="""
        Adds a data source to a project. A data source is a dataset
        in a supported format (check 'formats' section below).|n
        |n
        Currently, only local paths to sources are supported.|n
        Once added, a source is copied into project.|n
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
        Each dataset format has its own import options, which are passed
        after the '--' separator (see examples), pass '-- -h' for more info.|n
        |n
        Builtin formats: {}|n
        |n
        Examples:|n
        - Add a local directory with a VOC-like dataset:|n
        |s|s%(prog)s -f voc path/to/voc|n
        |n
        - Add a directory with a COCO dataset, use only a specific file:|n
        |s|s%(prog)s -f coco_instances -r anns/train.json path/to/coco|n
        |n
        - Add a local file with CVAT annotations, call it 'mysource'|n
        |s|s|s|sto the project in a specific place:|n
        |s|s%(prog)s -f cvat -n mysource -p project/path/ path/to/cvat.xml
        """.format(', '.join(builtins)),
        formatter_class=MultilineFormatter)

    parser.add_argument('_positionals', nargs=argparse.REMAINDER,
        help=argparse.SUPPRESS) # workaround for -- eaten by positionals
    parser.add_argument('url',
        help="URL to the source dataset. A path to a file or directory")
    parser.add_argument('-n', '--name',
        help="Name of the new source (default: generate automatically)")
    parser.add_argument('-f', '--format', required=True,
        help="Source dataset format")
    parser.add_argument('-r', '--path',
        help="A path relative to URL to the source data. Useful to specify "
            "a path to subset, subtask, or a specific file in URL.")
    parser.add_argument('--no-check', action='store_true',
        help="Don't try to read the source after importing")
    parser.add_argument('--no-cache', action='store_true',
        help="Don't put a copy into the project cache")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for extractor (pass '-- -h' for help). "
            "Must be specified after the main command arguments and after "
            "the '--' separator")
    parser.set_defaults(command=add_command)

    return parser

@scoped
def add_command(args):
    # Workaround. Required positionals consume positionals from the end
    args._positionals += join_cli_args(args, 'url', 'extra_args')

    has_sep = '--' in args._positionals
    if has_sep:
        pos = args._positionals.index('--')
    else:
        pos = 1
    args.url = (args._positionals[:pos] or [''])[0]
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

    fmt = args.format
    if fmt in env.importers:
        arg_parser = env.importers[fmt]
    elif fmt in env.extractors:
        arg_parser = env.extractors[fmt]
    else:
        raise CliException("Unknown format '%s'. A format can be added"
            " by providing an Extractor and Importer plugins" % fmt)

    extra_args = arg_parser.parse_cmdline(args.extra_args)

    name = args.name
    if name:
        if name in project.working_tree.sources:
            raise CliException("Source '%s' already exists" % name)
    else:
        name = generate_next_name(
            list(project.working_tree.sources) + os.listdir(),
            'source', sep='-', default='1')

    project.import_source(name, url=args.url, format=args.format,
        options=extra_args, no_cache=args.no_cache, rpath=args.path)
    on_error_do(project.remove_source, name, ignore_errors=True,
        kwargs={'force': True, 'keep_data': False})

    if not args.no_check:
        log.info("Checking the source...")
        project.working_tree.make_dataset(name)

    project.working_tree.save()

    log.info("Source '%s' with format '%s' has been added to the project",
        name, args.format)

    return 0

def build_remove_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Remove source from project",
        description="Remove a source from a project")

    parser.add_argument('names', nargs='+',
        help="Names of the sources to be removed")
    parser.add_argument('--force', action='store_true',
        help="Do not fail and stop on errors during removal")
    parser.add_argument('--keep-data', action='store_true',
        help="Do not remove source data from the working directory, remove "
            "only project metainfo.")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=remove_command)

    return parser

@scoped
def remove_command(args):
    project = scope_add(load_project(args.project_dir))

    if not args.names:
        raise CliException("Expected source name")

    for name in args.names:
        project.remove_source(name, force=args.force, keep_data=args.keep_data)
    project.working_tree.save()

    log.info("Sources '%s' have been removed from the project" % \
        ', '.join(args.names))

    return 0

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('name', nargs='?',
        help="Source name")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Show details")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=info_command)

    return parser

@scoped
def info_command(args):
    project = scope_add(load_project(args.project_dir))

    if args.name:
        source = project.working_tree.sources[args.name]
        print(source)
    else:
        for name, conf in project.working_tree.sources.items():
            print(name)
            if args.verbose:
                print(conf)

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(description="""
            Manipulate data sources inside of a project.|n
            |n
            A data source is a source of data for a project.
            The project combines multiple data sources into one dataset.
            The role of a data source is to provide dataset items - images
            and/or annotations.|n
            |n
            By default, the project to be operated on is searched for
            in the current directory. An additional '-p' argument can be
            passed to specify project location.
        """,
        formatter_class=MultilineFormatter)

    subparsers = parser.add_subparsers()
    add_subparser(subparsers, 'add', build_add_parser)
    add_subparser(subparsers, 'remove', build_remove_parser)
    add_subparser(subparsers, 'info', build_info_parser)

    return parser
