# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os

from datumaro.components.project import Environment
from datumaro.util import error_rollback

from ..util import MultilineFormatter, add_subparser
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
            The list of supported formats can be extended by project plugins.
            Check plugin section of developer guide for information about
            plugin implementation.|n
            |n
            Builtin formats: %s|n
            |n
            Examples:|n
            - Add a local directory with a VOC-like dataset:|n
            |s|sadd -f voc path/to/voc|n
            - Add a directory with a COCO dataset, use only a specific file:|n
            |s|sadd -f coco_instances path/to/voc -r anns/train.json|n
            - Add a local file with CVAT annotations, call it 'mysource'|n
            |s|s|s|sto the project in a specific place:|n
            |s|sadd -f cvat -n mysource -p project/path/ path/to/cvat.xml
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter)
    parser.add_argument('url',
        help="URL to the source dataset")
    parser.add_argument('-n', '--name',
        help="Name of the new source (default: generate automatically)")
    parser.add_argument('-f', '--format', required=True,
        help="Source dataset format")
    parser.add_argument('-r', '--path',
        help="A path relative to URL to the source data. Useful to specify "
            "a path to subset, subtask, or a specific file in URL.")
    parser.add_argument('--no-check', action='store_true',
        help="Skip source correctness checking")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for extractor (pass '-- -h' for help)")
    parser.set_defaults(command=add_command)

    return parser

@error_rollback('on_error', implicit=True)
def add_command(args):
    project = load_project(args.project_dir)

    name = args.name
    if name:
        if name in project.working_tree.sources:
            raise CliException("Source '%s' already exists" % name)
    else:
        name = generate_next_name(
            list(project.working_tree.sources) + os.listdir(),
            'source', sep='-', default='1')

    fmt = args.format
    if fmt in project.working_tree.env.importers:
        arg_parser = project.working_tree.env.importers[fmt]
    elif fmt in project.working_tree.env.extractors:
        arg_parser = project.working_tree.env.extractors[fmt]
    else:
        raise CliException("Unknown format '%s'. A format can be added"
            "by providing an Extractor and Importer plugins" % fmt)

    extra_args = {}
    if args.extra_args:
        if hasattr(arg_parser, 'parse_cmdline'):
            extra_args = arg_parser.parse_cmdline(args.extra_args)
        else:
            raise CliException("Format '%s' does not accept "
                "extra parameters" % fmt)

    project.import_source(name, url=args.url, format=args.format,
        options=extra_args)
    on_error.do(project.remove_source, name, force=True, keep_data=False,
        ignore_errors=True)

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
        help="Ignore possible errors during removal")
    parser.add_argument('--keep-data', action='store_true',
        help="Do not remove source data")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=remove_command)

    return parser

def remove_command(args):
    project = load_project(args.project_dir)

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

def info_command(args):
    project = load_project(args.project_dir)

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
