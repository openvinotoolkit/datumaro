# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.extractor_tfds import AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE
from datumaro.components.project import Environment
from datumaro.util.os_util import make_file_name

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import generate_next_file_name


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtin_writers = sorted(Environment().exporters)
    if TFDS_EXTRACTOR_AVAILABLE:
        available_datasets = ", ".join(f"tfds:{name}" for name in AVAILABLE_TFDS_DATASETS)
    else:
        available_datasets = "N/A (TensorFlow and/or TensorFlow Datasets " "are not installed)"

    parser = parser_ctor(
        help="Download a publicly available dataset",
        description="""
        Downloads a publicly available dataset and saves it in a given format.|n
        |n
        Currently, the only source of datasets is the TensorFlow Datasets
        library. Therefore, to use this command you must install TensorFlow &
        TFDS, which you can do as follows:|n
        |n
        |s|spip install datumaro[tf,tfds]|n
        |n
        Supported datasets: {}|n
        |n
        For information about the datasets, run "datum describe-downloads".
        More detailed information can be found in the TFDS Catalog:
        <https://www.tensorflow.org/datasets/catalog/overview>.|n
        |n
        Supported output formats: {}|n
        |n
        Examples:|n
        - Download the MNIST dataset:|n
        |s|s%(prog)s -i tfds:mnist -- --save-media|n
        |n
        - Download the VOC 2012 dataset, saving only the annotations in the COCO
          format into a specific directory:|n
        |s|s%(prog)s -i tfds:voc/2012 -f coco -o path/I/like/
        """.format(
            available_datasets, ", ".join(builtin_writers)
        ),
        formatter_class=MultilineFormatter,
    )

    parser.add_argument("-i", "--dataset-id", required=True, help="Which dataset to download")
    parser.add_argument(
        "-f", "--output-format", help="Output format (default: original format of the dataset)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="dst_dir",
        help="Directory to save output (default: a subdir in the current one)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in the save directory"
    )
    parser.add_argument("-s", "--subset", help="Save only the specified subset")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments for output format (pass '-- -h' for help). "
        "Must be specified after the main command arguments",
    )
    parser.set_defaults(command=download_command)

    return parser


def get_sensitive_args():
    return {
        download_command: ["dst_dir", "extra_args"],
    }


def download_command(args):
    env = Environment()

    if args.dataset_id.startswith("tfds:"):
        if TFDS_EXTRACTOR_AVAILABLE:
            tfds_ds_name = args.dataset_id[5:]
            tfds_ds = AVAILABLE_TFDS_DATASETS.get(tfds_ds_name)
            if tfds_ds:
                default_output_format = tfds_ds.metadata.default_output_format
                extractor_factory = tfds_ds.make_extractor
            else:
                raise CliException(f"Unsupported TFDS dataset '{tfds_ds_name}'")
        else:
            raise CliException(
                "TFDS datasets are not available, because TFDS and/or "
                "TensorFlow are not installed.\n"
                "You can install them with: pip install datumaro[tf,tfds]"
            )
    else:
        raise CliException(f"Unknown dataset ID '{args.dataset_id}'")

    output_format = args.output_format or default_output_format

    try:
        exporter = env.exporters[output_format]
    except KeyError:
        raise CliException("Exporter for format '%s' is not found" % output_format)
    extra_args = exporter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException(
                "Directory '%s' already exists " "(pass --overwrite to overwrite)" % dst_dir
            )
    else:
        dst_dir = generate_next_file_name(
            "%s-%s"
            % (
                make_file_name(args.dataset_id),
                make_file_name(output_format),
            )
        )
    dst_dir = osp.abspath(dst_dir)

    log.info("Downloading the dataset")
    extractor = extractor_factory()

    if args.subset:
        try:
            extractor = extractor.subsets()[args.subset]
        except KeyError:
            raise CliException("Subset '%s' is not present in the dataset" % args.subset)

    log.info("Exporting the dataset")
    exporter.convert(extractor, dst_dir, default_image_ext=".png", **extra_args)

    log.info("Dataset exported to '%s' as '%s'" % (dst_dir, output_format))
