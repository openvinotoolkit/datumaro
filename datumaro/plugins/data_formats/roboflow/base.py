# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import List, Optional, Union
from xml.etree import ElementTree

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.importer import ImportContext
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoImporterType, CocoTask
from datumaro.plugins.data_formats.voc.base import VocBase
from datumaro.plugins.data_formats.voc.format import VocImporterType, VocTask
from datumaro.plugins.data_formats.yolo.base import YoloUltralyticsBase
from datumaro.util.image import ImageMeta


class RoboflowCocoBase(_CocoBase):
    """
    Parses Roboflow COCO annotations written in the following format:
    https://cocodataset.org/#format-data
    """

    def __init__(
        self,
        path,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path,
            task=CocoTask.instances,
            coco_importer_type=CocoImporterType.roboflow,
            subset=subset,
            ctx=ctx,
        )


class RoboflowVocBase(VocBase):
    def __init__(
        self,
        path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(
            path,
            task=VocTask.voc_detection,
            voc_importer_type=VocImporterType.roboflow,
            subset=subset,
            ctx=ctx,
        )

    def _load_categories(self, path):
        cats = set()
        ann_files = [file for file in os.listdir(path) if file.endswith(".xml")]
        for ann_file in ann_files:
            xml_file = osp.join(path, ann_file)

            root = ElementTree.parse(xml_file).getroot()

            if root.tag != "annotation":
                continue

            for object_elem in root.iterfind("object"):
                cat_name = self._parse_field(object_elem, "name")
                cats.add(cat_name)

        label_categories = LabelCategories()
        for _, cat in enumerate(sorted(cats)):
            label_categories.add(cat)

        categories = {AnnotationType.label: label_categories}

        return categories

    def _load_subset_list(self, path):
        return [os.path.splitext(file)[0] for file in os.listdir(path) if file.endswith(".xml")]


class RoboflowYoloBase(YoloUltralyticsBase):
    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        urls: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(config_path, image_info, urls, **kwargs)
