# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from typing import Optional, Type, TypeVar

from defusedxml import ElementTree

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import SubsetBase
from datumaro.components.errors import InvalidAnnotationError, InvalidFieldError, MissingFieldError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, ImageFromFile
from datumaro.util.image import IMAGE_EXTENSIONS

T = TypeVar("T")


# class KaggleCsvBase(SubsetBase):
#     def __init__(
#         self,
#         path: str,
#         csv_path: str,
#         *,
#         subset: Optional[str] = None,
#         ctx: Optional[ImportContext] = None,
#     ):
#         if not subset:
#             subset = osp.splitext(osp.basename(path))[0]

#         super().__init__(subset=subset, ctx=ctx)

#         self._path = path
#         self._items = self._load_items(path)

#     def _load_items(self, path):
#         ann_infos = pd.read_csv(path)
#         cats = set()
#         ann_files = [file for file in os.listdir(path) if file.endswith(".xml")]
#         for ann_file in ann_files:
#             xml_file = osp.join(path, ann_file)

#             root = ElementTree.parse(xml_file).getroot()

#             if root.tag != "annotation":
#                 continue

#             for object_elem in root.iterfind("object"):
#                 cat_name = self._parse_field(object_elem, "name")
#                 cats.add(cat_name)

#         label_categories = LabelCategories()
#         for _, cat in enumerate(sorted(cats)):
#             label_categories.add(cat)

#         categories = {AnnotationType.label: label_categories}

#         return categories

#     def _load_subset_list(self, path):
#         return [os.path.splitext(file)[0] for file in os.listdir(path) if file.endswith(".xml")]


class KaggleRelaxedVocBase(SubsetBase):
    ann_extensions = ".xml"

    def __init__(
        self,
        img_path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, ctx=ctx)

        self._label_cat = LabelCategories()
        self._items = []
        self._size = None

        for img_filename in os.listdir(img_path):
            if not img_filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                continue
            item_id = os.path.splitext(img_filename)[0]

            img_file = os.path.join(img_path, img_filename)
            ann_file = os.path.join(ann_path, item_id + self.ann_extensions)

            if not os.path.isfile(ann_file):
                continue
            annotations = self._parse_annotations(img_file, ann_file)

            media = Image.from_file(path=img_file, size=self._size)

            self._items.append(
                DatasetItem(
                    id=item_id,
                    media=media,
                    annotations=annotations,
                )
            )
        self._categories = {AnnotationType.label: self._label_cat}

    def _parse_annotations(self, img_file: str, ann_file: str):
        root_elem = ElementTree.parse(ann_file).getroot()
        if root_elem.tag != "annotation":
            raise MissingFieldError("annotation")

        height = self._parse_field(root_elem, "size/height", int, required=False)
        width = self._parse_field(root_elem, "size/width", int, required=False)
        if height and width:
            self._size = (height, width)

        annotations = []
        for obj_id, object_elem in enumerate(root_elem.iterfind("object")):
            label_name = self._parse_field(object_elem, "name", str, required=True)

            bbox_elem = object_elem.find("bndbox")
            if not bbox_elem:
                raise MissingFieldError("bndbox")

            xmin = self._parse_field(bbox_elem, "xmin", float)
            xmax = self._parse_field(bbox_elem, "xmax", float)
            ymin = self._parse_field(bbox_elem, "ymin", float)
            ymax = self._parse_field(bbox_elem, "ymax", float)

            self._label_cat.add(label_name)
            label_id, _ = self._label_cat.find(label_name)
            annotations.append(
                Bbox(id=obj_id, label=label_id, x=xmin, y=ymin, w=xmax - xmin, h=ymax - ymin)
            )

        return annotations

    @staticmethod
    def _parse_field(root, xpath: str, cls: Type[T] = str, required: bool = True) -> Optional[T]:
        elem = root.find(xpath)
        if elem is None:
            if required:
                raise MissingFieldError(xpath)
            else:
                return None

        if cls is str:
            return elem.text

        try:
            return cls(elem.text)
        except Exception as e:
            raise InvalidFieldError(xpath) from e


class KaggleRelaxedYoloBase(KaggleRelaxedVocBase, SubsetBase):
    ann_extensions = ".txt"

    def __init__(
        self,
        img_path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(img_path=img_path, ann_path=ann_path, subset=subset, ctx=ctx)

    def _parse_annotations(self, img_file: str, ann_file: str):
        image = ImageFromFile(path=img_file)
        image_height, image_width = image.size

        lines = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []
        for obj_id, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                raise InvalidAnnotationError(
                    f"Unexpected field count {len(parts)} in the bbox description. "
                    "Expected 5 fields (label, xc, yc, w, h)."
                )
            label_id, xc, yc, w, h = parts
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            x = (xc - w * 0.5) * image_width
            y = (yc - h * 0.5) * image_height
            w *= image_width
            h *= image_height

            annotations.append(Bbox(id=obj_id, label=label_id, x=x, y=y, w=w, h=h))

        return annotations
