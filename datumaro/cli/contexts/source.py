# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.components.project import Environment
from datumaro.util import error_rollback

from ..util import CliException, MultilineFormatter, add_subparser
from ..util.project import generate_next_name, load_project


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    env = Environment()
    builtins = sorted(set(env.extractors) | set(env.importers))

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
            |s|sadd git://example.net/repo/path/to/coco/dir -f coco
        """ % ', '.join(builtins),
        formatter_class=MultilineFormatter)
    parser.add_argument('url',
        help="URL to the source dataset")
    parser.add_argument('-n', '--name',
        help="Name of the new source (default: generate automatically)")
    parser.add_argument('-f', '--format', required=True,
        help="Source dataset format")
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
        if name in project.sources:
            raise CliException("Source '%s' already exists" % name)
    else:
        name = generate_next_name(list(project.sources),
            'source', sep='-', default='1')

    fmt = args.format
    if fmt in project.env.importers:
        arg_parser = project.env.importers[fmt]
    elif fmt in project.env.extractors:
        arg_parser = project.env.extractors[fmt]
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

    project.sources.add(name, {
        'url': args.url,
        'format': args.format,
        'options': extra_args,
    })
    on_error.do(project.sources.remove, name, force=True, keep_data=False,
        ignore_errors=True)

    if not args.no_check:
        log.info("Checking the source...")
        project.sources.make_dataset(name)

    project.save()

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
        project.sources.remove(name, force=args.force, keep_data=args.keep_data)
    project.save()

    log.info("Sources '%s' have been removed from the project" % \
        ', '.join(args.names))

    return 0

def build_fetch_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Fetch updates for source into cache",
        description="Fetch updates for a source into the project cache")

    parser.add_argument('names', nargs='*',
        help="Names of sources (default: all)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=fetch_command)

    return parser

def fetch_command(args):
    project = load_project(args.project_dir)

    project.sources.fetch(args.names)

    return 0

def build_pull_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Fetch remote updates into cache",
        description="Fetch remote source updates into the project cache")

    parser.add_argument('names', nargs='*',
        help="Names of sources (default: all)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=pull_command)

    return parser

def pull_command(args):
    project = load_project(args.project_dir)

    project.sources.pull(args.names)

    return 0

def build_push_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Push local changes to remote",
        description="Push local source updates to a remote")

    parser.add_argument('names', nargs='+',
        help="Names of sources")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=push_command)

    return parser

def push_command(args):
    project = load_project(args.project_dir)

    project.sources.push(args.names)

    for source in args.names:
        stages = project.build_targets[source].stages
        stages[:] = stages[:1]
    project.save()

    return 0

def build_checkout_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Load source data from cache",
        description="Load source data from the project cache")

    parser.add_argument('names', nargs='*',
        help="Names of sources (default: all)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=checkout_command)

    return parser

def checkout_command(args):
    project = load_project(args.project_dir)

    project.sources.checkout(args.names)

    return 0

def build_update_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Update source revision to remote one",
        description="""
        Update source revision to remote one.|n
        |n
        To remove existing pipelines for the updated sources
        (start them from scratch), use the '--restart' parameter.|n
        |n
        A specific revision can be required by the '--rev' parameter.
        Otherwise, the latest remote version will be used.
        """)

    parser.add_argument('names', nargs='+',
        help="Names of sources to update")
    parser.add_argument('--rev',
        help="A revision to update the source to")
    parser.add_argument('--restart', action='store_true',
        help="Removes existing pipelines for these sources")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=update_command)

    return parser

def update_command(args):
    project = load_project(args.project_dir)

    for source in args.names:
        if source not in project.sources:
            raise KeyError("Unknown source '%s'" % source)

    project.sources.pull(args.names, rev=args.rev)
    for source in args.names:
        if args.restart:
            stages = project.build_targets[source].stages
            stages[:] = stages[:1]
        project.build_targets.build(source, reset=False, force=True)

    project.save()

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
        source = project.sources[args.name]
        print(source)
    else:
        for name, conf in project.sources.items():
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
    add_subparser(subparsers, 'fetch', build_fetch_parser)
    add_subparser(subparsers, 'checkout', build_checkout_parser)
    add_subparser(subparsers, 'pull', build_pull_parser)
    add_subparser(subparsers, 'push', build_push_parser)
    add_subparser(subparsers, 'update', build_push_parser)
    add_subparser(subparsers, 'info', build_info_parser)

    return parser
