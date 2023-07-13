# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import re
from enum import IntEnum
from typing import Dict


class YoloPath:
    DEFAULT_SUBSET_NAME = "train"
    SUBSET_NAMES = ["train", "valid"]
    RESERVED_CONFIG_KEYS = {"backup", "classes", "names"}

    @staticmethod
    def _parse_config(path: str) -> Dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            config_lines = f.readlines()

        config = {}

        for line in config_lines:
            match = re.match(r"^\s*(\w+)\s*=\s*(.+)$", line)
            if not match:
                continue

            key = match.group(1)
            value = match.group(2)
            config[key] = value

        return config


class YoloLoosePath:
    NAMES_FILE = "obj.names"


class YoloUltralyticsPath:
    META_FILE = "data.yaml"


class YoloFormatType(IntEnum):
    yolo_strict = 0
    yolo_loose = 1
    yolo_ultralytics = 2
