# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log

from datumaro.plugins.synthetic_images_plugin.image_generator import (
    ImageGenerator,
)

from ..util import MultilineFormatter


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Generate synthetic dataset",
        description="""
        Create a synthetic dataset whose elements have the specified shape and
        storing them in provided directory.|n
        To create 3-channel images, you should provide height and width for them.|n
        |n
        Examples:|n
        - Generate 300 3-channel synthetic images with H=224, W=256 and store to data_dir:|n
        |s|s%(prog)s -o data_dir -k 300 --shape 224 256|n
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-o', '--output-dir', required=True,
        help="Output directory to store generated dataset")
    parser.add_argument('-k', '--count', type=int, required=True,
        help="Number of images to be generated")
    parser.add_argument('--shape', nargs='+', type=int, required=True,
        help="Dimensions of data to be generated")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")

    parser.set_defaults(command=generate_command)

    return parser

def get_sensitive_args():
    return {
        generate_command: ['output_dir', 'count', 'shape']
    }

def generate_command(args):
    log.info("Generating dataset...")

    ImageGenerator(
        count=args.count,
        output_dir=args.output_dir,
        shape=args.shape,
        overwrite=args.overwrite
    ).generate_dataset()

    log.info("Results have been saved to '%s'", args.output_dir)

    return 0
