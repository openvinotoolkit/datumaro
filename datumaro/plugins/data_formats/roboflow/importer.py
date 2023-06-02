# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import defaultdict
from glob import glob
from io import TextIOWrapper
from typing import Any, Dict, List
from xml.etree import ElementTree

from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer


class RoboflowCocoImporter(Importer):
    @classmethod
    def detect(
        cls,
        context: FormatDetectionContext,
    ) -> FormatDetectionConfidence:
        context.require_file("*/_annotations.coco.json")
        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        subset_paths = glob(osp.join(path, "*", "_annotations.coco.json"), recursive=True)

        sources = []
        for subset_path in subset_paths:
            subset_name = osp.basename(osp.dirname(subset_path))
            sources.append(
                {"url": subset_path, "format": "roboflow_coco", "options": {"subset": subset_name}}
            )

        return sources


class RoboflowVocImporter(Importer):
    FORMAT = "roboflow_voc"
    FORMAT_EXT = ".xml"
    ANN_DIR_NAME = ""

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            with context.alternative():
                cls._check_ann_file(
                    context.require_file("**/" + cls.ANN_DIR_NAME + "*" + cls.FORMAT_EXT), context
                )

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(
            fpath, "Requirements for the annotation file of voc format"
        ) as fp:
            cls._check_ann_file_impl(fp)

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        root = ElementTree.parse(fp).getroot()

        if root.tag != "annotation":
            raise DatasetImportError("Roboflow VOC format xml file should have the annotation tag.")

        if not root.find("source/database").text == "roboflow.ai":
            raise DatasetImportError(
                "Roboflow VOC format xml file should have the source/database with `roboflow.ai`."
            )

        return True

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        def _filter_ann_file(fpath: str):
            try:
                with open(fpath, "r") as fp:
                    return cls._check_ann_file_impl(fp)
            except DatasetImportError:
                return False

        sources = cls._find_sources_recursive(
            path,
            ext=cls.FORMAT_EXT,
            extractor_name="",
            dirname=cls.ANN_DIR_NAME,
            file_filter=_filter_ann_file,
            filename="**/*",
            max_depth=1,
            recursive=True,
        )
        if len(sources) == 0:
            return []

        subsets = defaultdict(list)
        for source in sources:
            subset_name = osp.dirname(source["url"]).split("/")[-1]
            subsets[subset_name].append(source["url"])

        sources = [
            {
                "url": osp.join(path, subset),
                "format": cls.FORMAT,
                "options": {
                    "subset": subset,
                    # "urls": urls,
                },
            }
            for subset, urls in subsets.items()
        ]

        return sources


class RoboflowYoloImporter(RoboflowVocImporter):
    FORMAT = "roboflow_yolo"
    FORMAT_EXT = ".txt"
    ANN_DIR_NAME = "labels/"

    # @classmethod
    # def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
    #     with context.require_any():
    #         with context.alternative():
    #             cls._check_ann_file(context.require_file("**/labels/*" + cls.FORMAT_EXT), context)

    #     return FormatDetectionConfidence.MEDIUM

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

    # @classmethod
    # def find_sources(cls, path: str) -> List[Dict[str, Any]]:
    #     def _filter_ann_file(fpath: str):
    #         try:
    #             with open(fpath, "r") as fp:
    #                 return cls._check_ann_file_impl(fp)
    #         except DatasetImportError:
    #             return False

    #     sources = cls._find_sources_recursive(
    #         path,
    #         ext=".txt",
    #         extractor_name="",
    #         dirname="labels",
    #         file_filter=_filter_ann_file,
    #         filename="**/*",
    #         max_depth=1,
    #         recursive=True,
    #     )
    #     if len(sources) == 0:
    #         return []

    #     subsets = defaultdict(list)
    #     for source in sources:
    #         subset_name = osp.dirname(source["url"]).split("/")[-1]
    #         subsets[subset_name].append(source["url"])

    #     sources = [
    #         {
    #             "url": osp.join(path, subset),
    #             "format": "roboflow_yolo",
    #             "options": {
    #                 "subset": subset,
    #                 "urls": urls,
    #             },
    #         }
    #         for subset, urls in subsets.items()
    #     ]

    #     return sources


# class RoboflowYoloObbImporter(Importer):
#     raise NotImplementedError()


# class RoboflowDarknetImporter(Importer):
#     raise NotImplementedError()
