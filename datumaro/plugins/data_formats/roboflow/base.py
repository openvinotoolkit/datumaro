# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import List, Optional
from xml.etree import ElementTree

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from datumaro.plugins.data_formats.coco.base import _CocoBase
from datumaro.plugins.data_formats.coco.format import CocoImporterType, CocoTask
from datumaro.plugins.data_formats.voc.base import VocBase
from datumaro.plugins.data_formats.voc.format import VocImporterType, VocTask
from datumaro.plugins.data_formats.yolo.base import YoloStrictBase, YoloUltralyticsBase
from datumaro.util.image import IMAGE_EXTENSIONS
from datumaro.util.os_util import find_files


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


class RoboflowYoloBase(YoloUltralyticsBase, SubsetBase):
    def __init__(
        self,
        config_path: str,
        urls: Optional[List[str]] = None,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ) -> None:
        SubsetBase.__init__(self, subset=subset, ctx=ctx)

        rootpath = osp.dirname(config_path)

        # Init label categories
        label_categories = self._load_categories(osp.join(rootpath, self.META_FILE))
        self._categories = {AnnotationType.label: label_categories}

        # Parse dataset items
        def _get_fname(fpath: str) -> str:
            return osp.splitext(osp.basename(fpath))[0]

        img_files = {
            _get_fname(img_file): img_file
            for img_file in find_files(rootpath, IMAGE_EXTENSIONS, recursive=True, max_depth=2)
            if osp.split(osp.relpath(osp.dirname(img_file), rootpath))[0] == self._subset
        }

        for url in urls:
            try:
                fname = _get_fname(url)
                img = Image.from_file(path=img_files[fname])
                anns = YoloStrictBase._parse_annotations(
                    url,
                    img,
                    label_categories=label_categories,
                )
                self._items.append(
                    DatasetItem(id=fname, subset=self._subset, media=img, annotations=anns)
                )
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(fname, self._subset))
