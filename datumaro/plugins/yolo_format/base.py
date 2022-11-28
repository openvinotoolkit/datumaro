# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os.path as osp
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Type, TypeVar, Union

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, LabelCategories
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    UndeclaredLabelError,
)
from datumaro.components.dataset_base import DatasetItem, DatasetBase, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.image import DEFAULT_IMAGE_META_FILE_NAME, ImageMeta, load_image_meta_file
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file
from datumaro.util.os_util import split_path

from .format import YoloPath

T = TypeVar("T")


class YoloBase(SubsetBase):
    class Subset(DatasetBase):
        def __init__(self, name: str, parent: YoloBase):
            super().__init__()
            self._name = name
            self._parent = parent
            self.items: Dict[str, Union[str, DatasetItem]] = OrderedDict()

        def __iter__(self):
            for item_id in self.items:
                item = self._parent._get(item_id, self._name)
                if item is not None:
                    yield item

        def __len__(self):
            return len(self.items)

        def categories(self):
            return self._parent.categories()

    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        **kwargs,
    ) -> None:
        if not osp.isfile(config_path):
            raise DatasetImportError(f"Can't read dataset descriptor file '{config_path}'")

        super().__init__(**kwargs)

        rootpath = osp.dirname(config_path)
        self._path = rootpath

        assert image_info is None or isinstance(image_info, (str, dict))
        if image_info is None:
            image_info = osp.join(rootpath, DEFAULT_IMAGE_META_FILE_NAME)
            if not osp.isfile(image_info):
                image_info = {}
        if isinstance(image_info, str):
            image_info = load_image_meta_file(image_info)

        self._image_info = image_info

        config = self._parse_config(config_path)

        names_path = config.get("names")
        if not names_path:
            raise InvalidAnnotationError(f"Failed to parse names file path from config")

        # The original format is like this:
        #
        # classes = 2
        # train  = data/train.txt
        # valid  = data/test.txt
        # names = data/obj.names
        # backup = backup/
        #
        # To support more subset names, we disallow subsets
        # called 'classes', 'names' and 'backup'.
        subsets = {k: v for k, v in config.items() if k not in YoloPath.RESERVED_CONFIG_KEYS}

        for subset_name, list_path in subsets.items():
            list_path = osp.join(self._path, self.localize_path(list_path))
            if not osp.isfile(list_path):
                raise InvalidAnnotationError(f"Can't find '{subset_name}' subset list file")

            subset = YoloBase.Subset(subset_name, self)
            with open(list_path, "r", encoding="utf-8") as f:
                subset.items = OrderedDict(
                    (self.name_from_path(p), self.localize_path(p)) for p in f if p.strip()
                )
            subsets[subset_name] = subset

        self._subsets: Dict[str, YoloBase.Subset] = subsets

        self._categories = {
            AnnotationType.label: self._load_categories(
                osp.join(self._path, self.localize_path(names_path))
            )
        }

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

    @staticmethod
    def localize_path(path: str) -> str:
        """
        Removes the "data/" prefix from the path
        """

        path = osp.normpath(path.strip()).replace("\\", "/")
        default_base = "data/"
        if path.startswith(default_base):
            path = path[len(default_base) :]
        return path

    @classmethod
    def name_from_path(cls, path: str) -> str:
        """
        Obtains <image name> from the path like [data/]<subset>_obj/<image_name>.ext

        <image name> can be <a/b/c/filename>, so it is
        more involved than just calling "basename()".
        """

        path = cls.localize_path(path)

        parts = split_path(path)
        if 1 < len(parts) and not osp.isabs(path):
            path = osp.join(*parts[1:])  # pylint: disable=no-value-for-parameter

        return osp.splitext(path)[0]

    def _get(self, item_id: str, subset_name: str) -> Optional[DatasetItem]:
        subset = self._subsets[subset_name]
        item = subset.items[item_id]

        if isinstance(item, str):
            try:
                image_size = self._image_info.get(item_id)
                image = Image(path=osp.join(self._path, item), size=image_size)

                anno_path = osp.splitext(image.path)[0] + ".txt"
                annotations = self._parse_annotations(
                    anno_path, image, item_id=(item_id, subset_name)
                )

                item = DatasetItem(
                    id=item_id, subset=subset_name, media=image, annotations=annotations
                )
                subset.items[item_id] = item
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, subset_name))
                subset.items.pop(item_id)
                item = None

        return item

    @staticmethod
    def _parse_field(value: str, cls: Type[T], field_name: str) -> T:
        try:
            return cls(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {cls}"
            ) from e

    def _parse_annotations(
        self, anno_path: str, image: Image, *, item_id: Tuple[str, str]
    ) -> List[Annotation]:
        lines = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []

        if lines:
            # Use image info as late as possible to avoid unnecessary image loading
            if image.size is None:
                raise DatasetImportError(
                    f"Can't find image info for '{self.localize_path(image.path)}'"
                )
            image_height, image_width = image.size

        for line in lines:
            try:
                parts = line.split()
                if len(parts) != 5:
                    raise InvalidAnnotationError(
                        f"Unexpected field count {len(parts)} in the bbox description. "
                        "Expected 5 fields (label, xc, yc, w, h)."
                    )
                label_id, xc, yc, w, h = parts

                label_id = self._parse_field(label_id, int, "bbox label id")
                if label_id not in self._categories[AnnotationType.label]:
                    raise UndeclaredLabelError(str(label_id))

                w = self._parse_field(w, float, "bbox width")
                h = self._parse_field(h, float, "bbox height")
                x = self._parse_field(xc, float, "bbox center x") - w * 0.5
                y = self._parse_field(yc, float, "bbox center y") - h * 0.5
                annotations.append(
                    Bbox(
                        x * image_width,
                        y * image_height,
                        w * image_width,
                        h * image_height,
                        label=label_id,
                    )
                )
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=item_id)

        return annotations

    @staticmethod
    def _load_categories(names_path):
        if has_meta_file(osp.dirname(names_path)):
            return LabelCategories.from_iterable(parse_meta_file(osp.dirname(names_path)).keys())

        label_categories = LabelCategories()

        with open(names_path, "r", encoding="utf-8") as f:
            for label in f:
                label = label.strip()
                if label:
                    label_categories.add(label)

        return label_categories

    def __iter__(self):
        subsets = self._subsets
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for pbar, (subset_name, subset) in zip(pbars, subsets.items()):
            for item in pbar.iter(subset, desc=f"Parsing '{subset_name}'"):
                yield item

    def __len__(self):
        return sum(len(s) for s in self._subsets.values())

    def get_subset(self, name):
        return self._subsets[name]


class YoloImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".data", "yolo")
