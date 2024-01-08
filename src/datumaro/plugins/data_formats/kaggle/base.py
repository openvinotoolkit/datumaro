# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from typing import Dict, Optional, Type, TypeVar, Union

import pandas as pd
from defusedxml import ElementTree

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import SubsetBase
from datumaro.components.errors import InvalidAnnotationError, InvalidFieldError, MissingFieldError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, ImageFromFile
from datumaro.util.image import IMAGE_EXTENSIONS

T = TypeVar("T")


class KaggleImageCsvBase(SubsetBase):
    def __init__(
        self,
        path: str,
        img_path: str,
        csv_path: str,
        columns: Dict[str, str],
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        if not subset:
            subset = osp.splitext(osp.basename(path))[0]

        super().__init__(subset=subset, ctx=ctx)

        self._img_path = img_path
        self._columns = columns
        self._items, label_cat = self._load_items(csv_path, columns)
        self._categories = {AnnotationType.label: label_cat}

    def _load_items(self, csv_path: str, columns: Dict[str, Union[str, list]]):
        df = pd.read_csv(csv_path, header=None)

        indices = {}
        for key, field in columns.items():
            if key == "bbox":
                indices[key] = []
                for v in field:
                    indices[key].append(list(df.iloc[0]).index(v))
            else:
                indices[key] = list(df.iloc[0]).index(field)

        label_cat = LabelCategories()
        items = []
        for ind, row in df.iterrows():
            if ind == 0:
                continue

            data_info = list(row)
            media_name = data_info[indices["media"]]

            label_name = "default"
            if "label" in indices:
                label_name = str(data_info[indices["label"]])
            label_cat.add(label_name)
            label, _ = label_cat.find(label_name)

            # if "bbox" in indices:
            #     bbox = Bbox()

            if osp.exists(osp.join(self._img_path, media_name)):
                items.append(
                    DatasetItem(
                        id=osp.splitext(media_name)[0],
                        media=Image.from_file(path=osp.join(self._img_path, media_name)),
                        annotations=[
                            Label(label=label),
                            # Bbox(bbox=bbox),
                        ],
                    )
                )
        return items, label_cat


class KaggleVocBase(SubsetBase):
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


class KaggleYoloBase(KaggleVocBase, SubsetBase):
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
