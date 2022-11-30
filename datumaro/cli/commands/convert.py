# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.dataset import Dataset
from datumaro.components.project import Environment
from datumaro.util.os_util import make_file_name

from ..contexts.project import FilterModes
from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import generate_next_file_name


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtin_readers = sorted(set(Environment().importers) | set(Environment().extractors))
    builtin_writers = sorted(Environment().exporters)

    parser = parser_ctor(
        help="Convert an existing dataset to another format",
        description="""
        Converts a dataset from one format to another.
        You can add your own formats and do many more by creating a
        Datumaro project.|n
        |n
        This command serves as an alias for the "create", "import", and
        "export" commands, allowing to obtain the same results simpler
        and faster. Check descriptions of these commands for more info.|n
        |n
        Supported input formats: {}|n
        |n
        Supported output formats: {}|n
        |n
        Examples:|n
        - Export a dataset as a PASCAL VOC dataset, include images:|n
        |s|s%(prog)s -i src/path -f voc -- --save-images|n
        |n
        - Export a dataset as a COCO dataset to a specific directory:|n
        |s|s%(prog)s -i src/path -f coco -o path/I/like/
        """.format(
            ", ".join(builtin_readers), ", ".join(builtin_writers)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument(
        "-i",
        "--input-path",
        default=".",
        dest="source",
        help="Input dataset path (default: current dir)",
    )
    parser.add_argument(
        "-if", "--input-format", help="Input dataset format. Will try to detect, if not specified."
    )
    parser.add_argument("-f", "--output-format", required=True, help="Output format")
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: a subdir in the current one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument(
        "-e",
        "--filter",
        help='XML XPath filter expression for dataset items. Read "filter" '
        "command docs for more info",
    )
    parser.add_argument(
        "--filter-mode",
        default=FilterModes.i.name,
        type=FilterModes.parse,
        help="Filter mode, one of %s (default: %s)"
        % (", ".join(FilterModes.list_options()), "%(default)s"),
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for output format (pass '-- -h' for help). "
        "Must be specified after the main command arguments",
    )
    parser.set_defaults(command=convert_command)

    return parser


def get_sensitive_args():
    return {
        convert_command: ["source", "dst_dir", "extra_args"],
    }


def convert_command(args):
    env = Environment()

    try:
        exporter = env.exporters[args.output_format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % args.output_format)
    extra_args = exporter.parse_cmdline(args.extra_args)

    filter_args = FilterModes.make_filter_args(args.filter_mode)

    fmt = args.input_format
    if not args.input_format:
        matches = env.detect_dataset(args.source)
        if len(matches) == 0:
            log.error(
                "Failed to detect dataset format. "
                "Try to specify format with '-if/--input-format' parameter."
            )
            return 1
        elif len(matches) != 1:
            log.error(
                "Multiple formats match the dataset: %s. "
                "Try to specify format with '-if/--input-format' parameter.",
                ", ".join(matches),
            )
            return 2

        fmt = matches[0]
        log.info("Source dataset format detected as '%s'", args.input_format)

    source = osp.abspath(args.source)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name(
            "%s-%s" % (osp.basename(source), make_file_name(args.output_format))
        )
    dst_dir = osp.abspath(dst_dir)

    dataset = Dataset.import_from(source, fmt)

    log.info("Exporting the dataset")
    if args.filter:
        dataset = dataset.filter(args.filter, **filter_args)
    dataset.export(format=args.output_format, save_dir=dst_dir, **extra_args)

    log.info("Dataset exported to '%s' as '%s'" % (dst_dir, args.output_format))

    return 0
