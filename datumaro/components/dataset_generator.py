# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from abc import ABC, abstractmethod
from typing import Tuple

from datumaro.components.cli_plugin import CliPlugin


class DatasetGenerator(ABC, CliPlugin):
    """
    ImageGenerator generates synthetic data with provided shape.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-o",
            "--output-dir",
            type=osp.abspath,
            required=True,
            help="Path to the directory where dataset are saved",
        )
        parser.add_argument(
            "-k", "--count", type=int, required=True, help="Number of data to generate"
        )
        parser.add_argument(
            "--shape",
            nargs="+",
            type=int,
            required=True,
            help="Data shape. For example, for image with height = 256 and width = 224: --shape 256 224",
        )

        return parser

    def __init__(self, output_dir: str, count: int, shape: Tuple[int, int]):
        self._count = count
        self._shape = shape
        self._cpu_count = min(os.cpu_count(), self._count)

        self._output_dir = output_dir
        if not osp.exists(output_dir):
            os.makedirs(output_dir)

    @abstractmethod
    def generate_dataset(self) -> None:
        pass
