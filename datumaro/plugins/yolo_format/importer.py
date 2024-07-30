# Copyright (C) 2023 Intel Corporation
# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from os import path as osp
from typing import Any, Dict, List

import yaml

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.plugins.yolo_format.extractor import (
    YOLOv8Extractor,
    YOLOv8OrientedBoxesExtractor,
    YOLOv8PoseExtractor,
    YOLOv8SegmentationExtractor,
)
from datumaro.plugins.yolo_format.format import YOLOv8Path, YOLOv8PoseFormat


class YoloImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".data", "yolo")


class YOLOv8Importer(Importer):
    EXTRACTOR = YOLOv8Extractor

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--config-file",
            help="The name of the file to read dataset config from",
        )
        return parser

    @classmethod
    def _check_config_file(cls, context, config_file):
        with context.probe_text_file(
            config_file,
            f"must not have '{YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"*{YOLOv8Path.CONFIG_FILE_EXT}")
        sources = cls.find_sources_with_params(context.root_path)
        if not sources or len(sources) > 1:
            context.fail("Cannot choose config file")

        cls._check_config_file(context, osp.relpath(sources[0]["url"], context.root_path))

    @classmethod
    def find_sources_with_params(
        cls, path, config_file=None, **extra_params
    ) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(
            path, YOLOv8Path.CONFIG_FILE_EXT, cls.EXTRACTOR.NAME, max_depth=1
        )

        if config_file:
            return [source for source in sources if source["url"] == osp.join(path, config_file)]
        if len(sources) <= 1:
            return sources
        return [
            source
            for source in sources
            if source["url"] == osp.join(path, YOLOv8Path.DEFAULT_CONFIG_FILE)
        ]


class YOLOv8SegmentationImporter(YOLOv8Importer):
    EXTRACTOR = YOLOv8SegmentationExtractor


class YOLOv8OrientedBoxesImporter(YOLOv8Importer):
    EXTRACTOR = YOLOv8OrientedBoxesExtractor


class YOLOv8PoseImporter(YOLOv8Importer):
    EXTRACTOR = YOLOv8PoseExtractor

    @classmethod
    def _check_config_file(cls, context, config_file):
        with context.probe_text_file(
            config_file,
            f"must have '{YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if YOLOv8PoseFormat.KPT_SHAPE_FIELD_NAME not in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception
