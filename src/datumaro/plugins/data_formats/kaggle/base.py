# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import warnings
from typing import Dict, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from defusedxml import ElementTree

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
)
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetBase, SubsetBase
from datumaro.components.errors import InvalidAnnotationError, InvalidFieldError, MissingFieldError
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image, ImageFromFile
from datumaro.util.image import IMAGE_EXTENSIONS, load_image

T = TypeVar("T")


class KaggleImageCsvBase(DatasetBase):
    def __init__(
        self,
        path: str,
        ann_file: str,
        columns: Dict[str, str],
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(ctx=ctx)

        self._subset = subset

        self._path = path
        self._columns = columns

        self._items, label_cat = self._load_items(ann_file, columns)
        self._categories = {AnnotationType.label: label_cat}

    def _load_items(self, ann_file: str, columns: Dict[str, Union[str, list]]):
        df = pd.read_csv(ann_file, header=None, on_bad_lines="skip")

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
            id = osp.splitext(media_name)[0]

            if not media_name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                for ext in IMAGE_EXTENSIONS:
                    media_path = osp.join(self._path, media_name + ext)
                    if osp.exists(media_path):
                        break
            else:
                media_path = osp.join(self._path, media_name)

            if not osp.exists(media_path):
                warnings.warn(
                    f"'{media_path}' is not existed in the directory, "
                    f"so we skip to create an dataset item according to {row}."
                )
                continue

            annotations = []
            if "label" in indices:
                label_name = str(data_info[indices["label"]])
                label, cat = label_cat.find(label_name)

                if not cat:
                    label_cat.add(label_name)
                    label, _ = label_cat.find(label_name)

                annotations.append(Label(label=label))

            items.append(
                DatasetItem(
                    id=id,
                    subset=self._subset,
                    media=Image.from_file(path=media_path),
                    annotations=annotations,
                )
            )

        return items, label_cat

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class KaggleImageTxtBase(DatasetBase):
    def __init__(
        self,
        path: str,
        ann_file: str,
        columns: Dict[str, int],
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(ctx=ctx)

        self._subset = subset

        self._path = path
        self._columns = columns

        self._items, label_cat = self._load_items(ann_file, columns)
        self._categories = {AnnotationType.label: label_cat}

    def _load_items(self, ann_file: str, columns: Dict[str, Union[int, list]]):
        label_cat = LabelCategories()

        item_ids = []
        items = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split()
                media_name = line[columns["media"]]
                item_id = osp.splitext(media_name)[0]

                if item_id in item_ids:
                    warnings.warn(
                        f"There is duplicated '{id}' in {ann_file}, "
                        f"so we skip to create an dataset item according to {line}."
                    )
                    continue

                if not media_name.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                    for ext in IMAGE_EXTENSIONS:
                        media_path = osp.join(self._path, media_name + ext)
                        if osp.exists(media_path):
                            break
                else:
                    media_path = osp.join(self._path, media_name)

                if not osp.exists(media_path):
                    warnings.warn(
                        f"'{media_path}' is not existed in the directory, "
                        f"so we skip to create an dataset item according to {line}."
                    )
                    continue

                annotations = []
                if "label" in columns:
                    label_name = str(line[columns["label"]])
                    label, cat = label_cat.find(label_name)

                    if not cat:
                        label_cat.add(label_name)
                        label, _ = label_cat.find(label_name)

                    annotations.append(Label(label=label))

                item_ids.append(item_id)
                items.append(
                    DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image.from_file(path=media_path),
                        annotations=annotations,
                    )
                )

        return items, label_cat

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class KaggleImageMaskBase(DatasetBase):
    def __init__(
        self,
        path: str,
        mask_path: str,
        labelmap_file: Optional[str] = None,
        *,
        subset: Optional[str] = DEFAULT_SUBSET_NAME,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(ctx=ctx)

        self._subset = subset

        self._path = path
        self._mask_path = mask_path

        self._categories = self._load_categories(labelmap_file)
        self._items = self._load_items()

    def _load_categories(self, label_map_file: Optional[str]):
        label_map = dict()
        if not label_map_file:
            label_map["background"] = (0, 0, 0)
            label_map["object"] = (255, 255, 255)
        else:
            df = pd.read_csv(label_map_file)
            for _, row in df.iterrows():
                name = row[0]
                color = tuple([int(c) for c in row[1:]])
                label_map[name] = color

        label_categories = LabelCategories()
        for label in label_map:
            label_categories.add(label)

        categories = {}
        categories[AnnotationType.label] = label_categories

        colormap = {}
        for label_name, label_color in label_map.items():
            label_id = label_categories.find(label_name)[0]
            colormap[label_id] = label_color

        categories[AnnotationType.mask] = MaskCategories(colormap)

        return categories

    def _load_items(self):
        def _lazy_extract_mask(mask, c):
            return lambda: mask == c

        items = []
        for media_name in os.listdir(self._path):
            id = osp.splitext(media_name)[0]

            anns = []
            for mask_name in os.listdir(self._mask_path):
                if id in mask_name:
                    instances_mask = load_image(
                        osp.join(self._mask_path, mask_name), dtype=np.int32
                    )
                    label_ids = np.unique(instances_mask)
                    for label_id in label_ids:
                        anns.append(
                            Mask(
                                image=_lazy_extract_mask(instances_mask, label_id),
                                label=label_id,
                            )
                        )

            items.append(
                DatasetItem(
                    id=id,
                    subset=self._subset,
                    media=Image.from_file(path=osp.join(self._path, media_name)),
                    annotations=anns,
                )
            )

        return items

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items


class KaggleVocBase(SubsetBase):
    ann_extensions = ".xml"

    def __init__(
        self,
        path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(subset=subset, ctx=ctx)

        self._label_cat = LabelCategories()
        self._items = []
        self._size = None

        for img_filename in os.listdir(path):
            if not img_filename.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                continue
            item_id = os.path.splitext(img_filename)[0]

            img_file = os.path.join(path, img_filename)
            ann_file = os.path.join(ann_path, item_id + self.ann_extensions)

            annotations = (
                self._parse_annotations(img_file, ann_file) if os.path.isfile(ann_file) else []
            )

            media = Image.from_file(path=img_file, size=self._size)

            self._items.append(
                DatasetItem(
                    id=item_id,
                    subset=self._subset,
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
        path: str,
        ann_path: str,
        *,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(path=path, ann_path=ann_path, subset=subset, ctx=ctx)

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
            label_name, xc, yc, w, h = parts
            xc = float(xc)
            yc = float(yc)
            w = float(w)
            h = float(h)
            x = (xc - w * 0.5) * image_width
            y = (yc - h * 0.5) * image_height
            w *= image_width
            h *= image_height

            self._label_cat.add(label_name)
            label_id, _ = self._label_cat.find(label_name)

            annotations.append(Bbox(id=obj_id, label=label_id, x=x, y=y, w=w, h=h))

        return annotations
