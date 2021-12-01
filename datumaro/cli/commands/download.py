# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
import os
import os.path as osp

from datumaro.components.extractor_tfds import (
    AVAILABLE_TFDS_DATASETS, TFDS_EXTRACTOR_AVAILABLE, make_tfds_extractor,
)
from datumaro.components.project import Environment
from datumaro.util.os_util import make_file_name

from ..util import MultilineFormatter
from ..util.errors import CliException
from ..util.project import generate_next_file_name


def build_parser(parser_ctor=argparse.ArgumentParser):
    builtin_writers = sorted(Environment().converters)
    if TFDS_EXTRACTOR_AVAILABLE:
        available_datasets = ", ".join(
            f'tfds:{name}' for name in AVAILABLE_TFDS_DATASETS)
    else:
        available_datasets = "N/A (TensorFlow and/or TensorFlow Datasets " \
            "are not installed)"

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
        For information about the datasets, see the TFDS Catalog:
        <https://www.tensorflow.org/datasets/catalog/overview>.|n
        |n
        Supported output formats: {}|n
        |n
        Examples:|n
        - Download the MNIST dataset, saving it in the ImageNet text format:|n
        |s|s%(prog)s -i tfds:mnist -f imagenet_txt -- --save-images|n
        |n
        - Download the VOC 2012 dataset, saving only the annotations in the COCO
          format into a specific directory:|n
        |s|s%(prog)s -i tfds:voc/2012 -f coco -o path/I/like/
        """.format(available_datasets, ', '.join(builtin_writers)),
        formatter_class=MultilineFormatter)

    parser.add_argument('-i', '--dataset-id', required=True,
        help="Which dataset to download")
    parser.add_argument('-f', '--output-format', required=True,
        help="Output format")
    parser.add_argument('-o', '--output-dir', dest='dst_dir',
        help="Directory to save output (default: a subdir in the current one)")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
        help="Additional arguments for output format (pass '-- -h' for help). "
            "Must be specified after the main command arguments")
    parser.set_defaults(command=download_command)

    return parser

def get_sensitive_args():
    return {
        download_command: ['dst_dir', 'extra_args'],
    }

def download_command(args):
    env = Environment()

    try:
        converter = env.converters[args.output_format]
    except KeyError:
        raise CliException("Converter for format '%s' is not found" %
            args.output_format)
    extra_args = converter.parse_cmdline(args.extra_args)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('%s-%s' % (
            make_file_name(args.dataset_id),
            make_file_name(args.output_format),
        ))
    dst_dir = osp.abspath(dst_dir)

    log.info("Downloading the dataset")
    if TFDS_EXTRACTOR_AVAILABLE and args.dataset_id.startswith('tfds:'):
        tfds_ds_name = args.dataset_id[5:]
        if tfds_ds_name in AVAILABLE_TFDS_DATASETS:
            extractor = make_tfds_extractor(tfds_ds_name)
        else:
            raise CliException(f"Unsupported TFDS dataset '{tfds_ds_name}'")
    else:
        raise CliException(f"Unknown dataset ID '{args.dataset_id}'")

    log.info("Exporting the dataset")
    converter.convert(extractor, dst_dir,
        default_image_ext='.png', **extra_args)

    log.info("Dataset exported to '%s' as '%s'" % \
        (dst_dir, args.output_format))
