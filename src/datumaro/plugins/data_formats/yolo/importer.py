# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import defaultdict
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Type

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.merge.extractor_merger import ExtractorMerger
from datumaro.util.os_util import extract_subset_name_from_parent

from .format import YoloFormatType, YoloLoosePath, YoloPath, YoloUltralyticsPath


class _YoloStrictImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict]:
        found = cls._find_sources_recursive(path, ".data", YoloFormatType.yolo_strict.name)
        if len(found) == 0:
            return []

        config_path = found[0]["url"]
        config = YoloPath._parse_config(config_path)
        subsets = [k for k in config if k not in YoloPath.RESERVED_CONFIG_KEYS]
        return [
            {
                "url": config_path,
                "format": YoloFormatType.yolo_strict.name,
                "options": {"subset": subset},
            }
            for subset in subsets
        ]


class _YoloLooseImporter(Importer):
    META_FILE = YoloLoosePath.NAMES_FILE
    FORMAT = YoloFormatType.yolo_loose.name

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        context.require_file(cls.META_FILE)

        with context.require_any():
            with context.alternative():
                cls._check_ann_file(context.require_file("[Aa]nnotations/**/*.txt"), context)
            with context.alternative():
                cls._check_ann_file(context.require_file("[Ll]abels/**/*.txt"), context)

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(
            fpath, "Requirements for the annotation file of yolo format"
        ) as fp:
            cls._check_ann_file_impl(fp)

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        for line in fp:
            fields = line.rstrip("\n").split(" ")
            if len(fields) != 5:
                raise DatasetImportError(
                    f"Yolo format txt file should have 5 fields for each line, "
                    f"but the read line has {len(fields)} fields: fields={fields}."
                )

            for field in fields:
                if not field.replace(".", "").isdigit():
                    raise DatasetImportError(f"Each field should be a number but fields={fields}.")

            # Check the first line only
            return True

        raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def _find_loose(cls, path: str, dirname: str) -> List[Dict[str, Any]]:
        def _filter_ann_file(fpath: str):
            try:
                with open(fpath, "r") as fp:
                    return cls._check_ann_file_impl(fp)
            except DatasetImportError:
                return False

        sources = cls._find_sources_recursive(
            path,
            ext=".txt",
            extractor_name="",
            dirname=dirname,
            file_filter=_filter_ann_file,
            filename="**/*",
            max_depth=1,
            recursive=True,
        )
        if len(sources) == 0:
            return []

        subsets = defaultdict(list)

        for source in sources:
            subsets[extract_subset_name_from_parent(source["url"], path)].append(source["url"])

        sources = [
            {
                "url": osp.join(path),
                "format": cls.FORMAT,
                "options": {
                    "subset": subset,
                    "urls": urls,
                },
            }
            for subset, urls in subsets.items()
        ]
        return sources

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        # Check obj.names first
        filename, ext = osp.splitext(cls.META_FILE)
        sources = cls._find_sources_recursive(
            path,
            ext=ext,
            extractor_name="",
            dirname="",
            filename=filename,
            max_depth=1,
            recursive=False,
        )
        if len(sources) == 0:
            return []

        # TODO: From Python >= 3.8, we can use
        # "if (sources := cls._find_strict(path)): return sources"
        sources = cls._find_loose(path, "[Aa]nnotations")
        if sources:
            return sources

        sources = cls._find_loose(path, "[Ll]abels")
        if sources:
            return sources

        return []

    @property
    def can_stream(self) -> bool:
        return True


class _YoloUltralyticsImporter(_YoloLooseImporter):
    META_FILE = YoloUltralyticsPath.META_FILE
    FORMAT = YoloFormatType.yolo_ultralytics.name


class YoloImporter(Importer):
    SUB_IMPORTERS: Dict[YoloFormatType, Importer] = {
        YoloFormatType.yolo_strict: _YoloStrictImporter,
        YoloFormatType.yolo_loose: _YoloLooseImporter,
        YoloFormatType.yolo_ultralytics: _YoloUltralyticsImporter,
    }

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            for importer_cls in cls.SUB_IMPORTERS.values():
                with context.alternative():
                    return importer_cls.detect(context)

        context.fail("Any yolo format is not detected.")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        for importer_cls in cls.SUB_IMPORTERS.values():
            # TODO: From Python >= 3.8, we can use
            # "if (sources := importer_cls.find_sources(path)): return sources"
            sources = importer_cls.find_sources(path)
            if sources:
                return sources

        return []

    def get_extractor_merger(self) -> Optional[Type[ExtractorMerger]]:
        return ExtractorMerger
