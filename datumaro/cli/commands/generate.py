# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from shutil import rmtree
import argparse
import logging as log
import os
import os.path as osp

from datumaro.cli.util.errors import CliException
from datumaro.plugins.synthetic_data import ImageGenerator

from ..util import MultilineFormatter


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Generate synthetic dataset",
        description="""
        Creates a synthetic dataset with elements of the specified shape and
        saves it in the provided directory.|n
        To create 3-channel images, you should provide height and width for them.|n
        |n
        Examples:|n
        - Generate 300 3-channel synthetic images with H=224, W=256 and store to data_dir:|n
        |s|s%(prog)s -o data_dir -k 300 --shape 224 256|n
        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-t', '--type', required=True, choices=['image'],
        help="Specify type of data to generate")
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
        generate_command: ['type', 'output_dir', 'count', 'shape']
    }

def generate_command(args):
    log.info("Generating dataset...")
    output_dir = args.output_dir

    if osp.isdir(output_dir) and os.listdir(output_dir):
        if args.overwrite:
            rmtree(output_dir)
            os.mkdir(output_dir)
        else:
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to overwrite)" % output_dir)

    if args.type == 'image':
        ImageGenerator(
            count=args.count,
            output_dir=output_dir,
            shape=args.shape
        ).generate_dataset()
    else:
        raise NotImplementedError(f'Data type: {args.type} is not supported')

    log.info("Results have been saved to '%s'", args.output_dir)

    return 0
