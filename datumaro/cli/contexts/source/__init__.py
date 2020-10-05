
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.project import \
    PROJECT_DEFAULT_CONFIG as DEFAULT_CONFIG
from datumaro.components.project import Environment, Project

from ...util import (CliException, MultilineFormatter, add_subparser,
    make_file_name)
from ...util.project import generate_next_file_name, load_project


def build_add_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().extractors.items)

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-n', '--name', default=None,
        help="Name of the new source")
    base_parser.add_argument('-f', '--format', required=True,
        help="Source dataset format")
    base_parser.add_argument('--skip-check', action='store_true',
        help="Skip source checking")
    base_parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")

    parser = parser_ctor(help="Add data source to project",
        description="""
            Adds a data source to a project. The source can be:|n
            - a dataset in a supported format (check 'formats' section below)|n
            - a Datumaro project|n
            |n
            The source can be either a local directory or a remote
            git repository. Each source type has its own parameters, which can
            be checked by:|n
            '%s'.|n
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
            definition script to <project_dir>/.datumaro/extractors.|n
            |n
            List of builtin source formats: %s|n
            |n
            Examples:|n
            - Add a local directory with VOC-like dataset:|n
            |s|sadd path path/to/voc -f voc_detection|n
            - Add a local file with CVAT annotations, call it 'mysource'|n
            |s|s|s|sto the project somewhere else:|n
            |s|sadd path path/to/cvat.xml -f cvat -n mysource -p somewhere/else/
        """ % ('%(prog)s SOURCE_TYPE --help', ', '.join(builtins)),
        formatter_class=MultilineFormatter,
        add_help=False)
    parser.set_defaults(command=add_command)

    sp = parser.add_subparsers(dest='source_type', metavar='SOURCE_TYPE',
        help="The type of the data source "
            "(call '%s SOURCE_TYPE --help' for more info)" % parser.prog)

    dir_parser = sp.add_parser('path', help="Add local path as source",
        parents=[base_parser])
    dir_parser.add_argument('url',
        help="Path to the source")
    dir_parser.add_argument('--copy', action='store_true',
        help="Copy the dataset instead of saving source links")

    repo_parser = sp.add_parser('git', help="Add git repository as source",
        parents=[base_parser])
    repo_parser.add_argument('url',
        help="URL of the source git repository")
    repo_parser.add_argument('-b', '--branch', default='master',
        help="Branch of the source repository (default: %(default)s)")
    repo_parser.add_argument('--checkout', action='store_true',
        help="Do branch checkout")

    # NOTE: add common parameters to the parent help output
    # the other way could be to use parse_known_args()
    display_parser = argparse.ArgumentParser(
        parents=[base_parser, parser],
        prog=parser.prog, usage="%(prog)s [-h] SOURCE_TYPE ...",
        description=parser.description, formatter_class=MultilineFormatter)
    class HelpAction(argparse._HelpAction):
        def __call__(self, parser, namespace, values, option_string=None):
            display_parser.print_help()
            parser.exit()

    parser.add_argument('-h', '--help', action=HelpAction,
        help='show this help message and exit')

    # TODO: needed distinction on how to add an extractor or a remote source

    return parser

def add_command(args):
    project = load_project(args.project_dir)

    if args.source_type == 'git':
        name = args.name
        if name is None:
            name = osp.splitext(osp.basename(args.url))[0]

        if project.env.git.has_submodule(name):
            raise CliException("Git submodule '%s' already exists" % name)

        try:
            project.get_source(name)
            raise CliException("Source '%s' already exists" % name)
        except KeyError:
            pass

        rel_local_dir = project.local_source_dir(name)
        local_dir = osp.join(project.config.project_dir, rel_local_dir)
        url = args.url
        project.env.git.create_submodule(name, local_dir,
            url=url, branch=args.branch, no_checkout=not args.checkout)
    elif args.source_type == 'path':
        url = osp.abspath(args.url)
        if not osp.exists(url):
            raise CliException("Source path '%s' does not exist" % url)

        name = args.name
        if name is None:
            name = osp.splitext(osp.basename(url))[0]

        if project.env.git.has_submodule(name):
            raise CliException("Git submodule '%s' already exists" % name)

        try:
            project.get_source(name)
            raise CliException("Source '%s' already exists" % name)
        except KeyError:
            pass

        rel_local_dir = project.local_source_dir(name)
        local_dir = osp.join(project.config.project_dir, rel_local_dir)

        if args.copy:
            log.info("Copying from '%s' to '%s'" % (url, local_dir))
            if osp.isdir(url):
                # copytree requires destination dir not to exist
                shutil.copytree(url, local_dir)
                url = rel_local_dir
            elif osp.isfile(url):
                os.makedirs(local_dir)
                shutil.copy2(url, local_dir)
                url = osp.join(rel_local_dir, osp.basename(url))
            else:
                raise Exception("Expected file or directory")
        else:
            os.makedirs(local_dir)

    project.add_source(name, { 'url': url, 'format': args.format })

    if not args.skip_check:
        log.info("Checking the source...")
        try:
            project.make_source_project(name).make_dataset()
        except Exception:
            shutil.rmtree(local_dir, ignore_errors=True)
            raise

    project.save()

    log.info("Source '%s' has been added to the project, location: '%s'" \
        % (name, rel_local_dir))

    return 0

def build_import_parser(parser_ctor=argparse.ArgumentParser):
    builtins = sorted(Environment().importers.items)

    parser = parser_ctor(help="Create project from existing dataset",
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
    if not args.format:
        if args.extra_args:
            raise CliException("Extra args can not be used without format")

        log.info("Trying to detect dataset format...")

        matches = []
        for format_name in env.importers.items:
            log.debug("Checking '%s' format...", format_name)
            importer = env.make_importer(format_name)
            try:
                match = importer.detect(args.source)
                if match:
                    log.debug("format matched")
                    matches.append((format_name, importer))
            except NotImplementedError:
                log.debug("Format '%s' does not support auto detection.",
                    format_name)

        if len(matches) == 0:
            log.error("Failed to detect dataset format automatically. "
                "Try to specify format with '-f/--format' parameter.")
            return 1
        elif len(matches) != 1:
            log.error("Multiple formats match the dataset: %s. "
                "Try to specify format with '-f/--format' parameter.",
                ', '.join(m[0] for m in matches))
            return 2

        format_name, importer = matches[0]
        args.format = format_name
    else:
        try:
            importer = env.make_importer(args.format)
            if hasattr(importer, 'from_cmdline'):
                extra_args = importer.from_cmdline(args.extra_args)
        except KeyError:
            raise CliException("Importer for format '%s' is not found" % \
                args.format)

    log.info("Importing project as '%s'" % args.format)

    source = osp.abspath(args.source)
    project = importer(source, **extra_args)
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
    try:
        project.get_source(name)
    except KeyError:
        if not args.force:
            raise CliException("Source '%s' does not exist" % name)

    if project.env.git.has_submodule(name):
        if args.force:
            log.warning("Forcefully removing the '%s' source..." % name)

        project.env.git.remove_submodule(name, force=args.force)

    source_dir = osp.join(project.config.project_dir,
        project.local_source_dir(name))
    project.remove_source(name)
    project.save()

    if not args.keep_data:
        shutil.rmtree(source_dir, ignore_errors=True)

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
        source = project.get_source(args.name)
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
