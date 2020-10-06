
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=unused-import

from ..contexts.source import build_import_parser as build_parser


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
        if args.overwrite:
            shutil.rmtree(project_env_dir, ignore_errors=True)
        else:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % project_env_dir)

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
