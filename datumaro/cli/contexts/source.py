# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import shutil

from datumaro.components.project import Environment

from ..util import CliException, MultilineFormatter, add_subparser
from ..util.project import generate_next_name, load_project


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().extractors.items)

    parser = parser_ctor(help="Add data source to project",
        description="""
            Adds a data source to a project. The source can be:|n
            - a dataset in a supported format (check 'formats' section below)|n
            - a Datumaro project|n
            |n
            The source can be a local path or a remote link.|n
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
            In Datumaro dataset formats are supported by Extractor-s.
            An Extractor produces a list of dataset items corresponding
            to the dataset. It is possible to add a custom Extractor.
            To do this, you need to put an Extractor
            definition script to <project_dir>/.datumaro/plugins.|n
            |n
            List of builtin source formats: %s|n
            |n
            Examples:|n
            - Add a local directory with VOC-like dataset:|n
            |s|sadd path/to/voc -f voc|n
            - Add a local file with CVAT annotations, call it 'mysource'|n
            |s|s|s|sto the project somewhere else:|n
            |s|sadd path/to/cvat.xml -f cvat -n mysource -p somewhere/|n
            - Add a remote link to a COCO-like dataset:|n
            |s|sadd git://example.net/repo/path/to/coco/dir -f coco|n
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter,
        add_help=False)
    parser.add_argument('url',
        help="Path to the source dataset")
    parser.add_argument('-n', '--name',
        help="Name of the new source")
    parser.add_argument('-f', '--format', required=True,
        help="Source dataset format")
    parser.add_argument('--no-check', action='store_true',
        help="Skip source correctness checking")
    parser.add_argument('--no-pull', action='store_true',
        help="Do not pull the source")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for output format (pass '-- -h' for help)")
    parser.set_defaults(command=add_command)

    return parser

def add_command(args):
    project = load_project(args.project_dir)

    name = args.name
    if name is None:
        name = generate_next_name(list(project.sources), 'source', sep='-')

    extra_args = {}
    if args.extra_args:
        try:
            extractor = project.env.extractors[args.format]
        except KeyError:
            raise CliException("Extractor for format '%s' is not found" % \
                args.format)
        extra_args = extractor.from_cmdline(args.extra_args)

    project.sources.add(name, {
        'url': args.url,
        'format': args.format,
        'options': extra_args,
    })

    if not args.no_pull:
        log.info("Pulling the source...")
        project.sources.pull(name)

    if not (args.no_check or args.no_pull):
        log.info("Checking the source...")
        project.sources.make_dataset(name)

    project.save()

    log.info("Source '%s' has been added to the project" % name)

    return 0

def build_remove_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Remove source from project",
        description="Remove a source from a project.")

    parser.add_argument('-n', '--name', required=True,
        help="Name of the source to be removed")
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

    name = args.name
    if not name:
        raise CliException("Expected source name")

    project.sources.remove(name, force=args.force, keep_data=args.keep_data)
    project.save()

    log.info("Source '%s' has been removed from the project" % name)

    return 0

def build_info_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor()

    parser.add_argument('-n', '--name',
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
        source = project.sources[args.name]
        print(source)
    else:
        for name, conf in project.config.sources.items():
            print(name)
            if args.verbose:
                print(dict(conf))

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
