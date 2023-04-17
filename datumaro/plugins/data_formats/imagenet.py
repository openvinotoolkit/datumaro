# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import MediaTypeError
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer, with_subset_dirs
from datumaro.components.media import Image
from datumaro.util.definitions import SUBSET_NAME_BLACKLIST
from datumaro.util.image import find_images


class ImagenetPath:
    IMAGE_DIR_NO_LABEL = "no_label"
    SEP_TOKEN = ":"


class ImagenetBase(SubsetBase):
    def __init__(self, path, subset=None):
        assert osp.isdir(path), path
        super().__init__(subset=subset)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_cat = LabelCategories()
        for dirname in sorted(os.listdir(path)):
            if dirname != ImagenetPath.IMAGE_DIR_NO_LABEL:
                label_cat.add(dirname)
        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}

        # Images should be in root/label_dir/*.img and root/*.img is not allowed.
        # => max_depth=1, min_depth=1
        for image_path in find_images(path, recursive=True, max_depth=1, min_depth=1):
            label = osp.basename(osp.dirname(image_path))
            image_name = osp.splitext(osp.basename(image_path))[0]

            item_id = label + ImagenetPath.SEP_TOKEN + image_name
            item = items.get(item_id)
            if item is None:
                item = DatasetItem(
                    id=item_id, subset=self._subset, media=Image.from_file(path=image_path)
                )
                items[item_id] = item
            annotations = item.annotations

            if label != ImagenetPath.IMAGE_DIR_NO_LABEL:
                label = self._categories[AnnotationType.label].find(label)[0]
                annotations.append(Label(label=label))

        return items


class ImagenetImporter(Importer):
    """TorchVision's ImageFolder style importer.
    For example, it imports the following directory structure.

    .. code-block:: text

        root
        ├── label_0
        │   ├── label_0_1.jpg
        │   └── label_0_2.jpg
        └── label_1
            └── label_1_1.jpg

    """

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        # Images must not be under a directory whose name is blacklisted.
        for dname in os.listdir(context.root_path):
            dpath = osp.join(context.root_path, dname)
            if osp.isdir(dpath) and dname.lower() in SUBSET_NAME_BLACKLIST:
                context.fail(
                    f"{dname} is found in {context.root_path}. "
                    "However, Images must not be under a directory whose name is blacklisted "
                    f"(SUBSET_NAME_BLACKLIST={SUBSET_NAME_BLACKLIST})."
                )

        return super().detect(context)

    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []

        # Images should be in root/label_dir/*.img and root/*.img is not allowed.
        # => max_depth=1, min_depth=1
        for _ in find_images(path, recursive=True, max_depth=1, min_depth=1):
            return [{"url": path, "format": ImagenetBase.NAME}]

        return []


@with_subset_dirs
class ImagenetWithSubsetDirsImporter(ImagenetImporter):
    """TorchVision ImageFolder style importer.
    For example, it imports the following directory structure.

    .. code-block::

        root
        ├── train
        │   ├── label_0
        │   │   ├── label_0_1.jpg
        │   │   └── label_0_2.jpg
        │   └── label_1
        │       └── label_1_1.jpg
        ├── val
        │   ├── label_0
        │   │   ├── label_0_1.jpg
        │   │   └── label_0_2.jpg
        │   └── label_1
        │       └── label_1_1.jpg
        └── test
            ├── label_0
            │   ├── label_0_1.jpg
            │   └── label_0_2.jpg
            └── label_1
                └── label_1_1.jpg

    Then, it will have three subsets: train, val, and test and they have label_0 and label_1 labels.
    """


class ImagenetExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".jpg"
    USE_SUBSET_DIRS = False

    def apply(self):
        def _get_name(item: DatasetItem) -> str:
            id_parts = item.id.split(ImagenetPath.SEP_TOKEN)

            if len(id_parts) == 1:
                # e.g. item.id = my_img_1
                return item.id
            else:
                # e.g. item.id = label_1:my_img_1
                return "_".join(id_parts[1:])  # ":" is not allowed in windows

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        if 1 < len(self._extractor.subsets()) and not self.USE_SUBSET_DIRS:
            log.warning(
                f"There are more than one subset in the dataset ({len(self._extractor.subsets())}). "
                "However, ImageNet format exports all dataset items into the same directory. "
                "Therefore, subset information will be lost. To prevent it, please use ImagenetWithSubsetDirsExporter. "
                'For example, dataset.export("<path/to/output>", format="imagenet_with_subset_dirs").'
            )

        root_dir = self._save_dir
        extractor = self._extractor
        labels = {}
        for item in self._extractor:
            file_name = _get_name(item)
            labels = set(p.label for p in item.annotations if p.type == AnnotationType.label)

            for label in labels:
                label_name = extractor.categories()[AnnotationType.label][label].name
                self._save_image(
                    item,
                    subdir=osp.join(root_dir, item.subset, label_name)
                    if self.USE_SUBSET_DIRS
                    else osp.join(root_dir, label_name),
                    name=file_name,
                )

            if not labels:
                self._save_image(
                    item,
                    subdir=osp.join(root_dir, item.subset, ImagenetPath.IMAGE_DIR_NO_LABEL)
                    if self.USE_SUBSET_DIRS
                    else osp.join(root_dir, ImagenetPath.IMAGE_DIR_NO_LABEL),
                    name=file_name,
                )


class ImagenetWithSubsetDirsExporter(ImagenetExporter):
    USE_SUBSET_DIRS = True
