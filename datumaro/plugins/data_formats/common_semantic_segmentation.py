# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer, with_subset_dirs
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import generate_colormap, lazy_mask
from datumaro.util.meta_file_util import DATASET_META_FILE, is_meta_file, parse_meta_file


class CommonSemanticSegmentationPath:
    MASKS_DIR = "masks"
    IMAGES_DIR = "images"


def make_categories(label_map=None):
    if label_map is None:
        return {}

    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors:  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = {label_id(name): (desc[0], desc[1], desc[2]) for name, desc in label_map.items()}
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap  # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories
    return categories


class CommonSemanticSegmentationBase(SubsetBase):
    def __init__(
        self,
        path,
        subset=None,
        image_prefix="",
        mask_prefix="",
    ):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__(subset=subset)

        self._image_prefix = image_prefix
        self._mask_prefix = mask_prefix

        meta_file = glob.glob(osp.join(path, "**", DATASET_META_FILE), recursive=True)
        if is_meta_file(meta_file[0]):
            self._root_dir = osp.dirname(meta_file[0])

            label_map = parse_meta_file(meta_file[0])
            self._categories = make_categories(label_map)
        else:
            raise DatasetImportError("Dataset meta info file was not found in %s" % path)

        self._items = list(self._load_items().values())

    def _load_items(self):
        items = {}

        image_dir = osp.join(self._root_dir, CommonSemanticSegmentationPath.IMAGES_DIR)

        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/")[
                    len(self._image_prefix) :
                ]: p
                for p in find_images(image_dir, recursive=True)
                if osp.basename(p).startswith(self._image_prefix)
            }
        else:
            images = {}

        mask_dir = osp.join(self._root_dir, CommonSemanticSegmentationPath.MASKS_DIR)
        masks = [
            mask_path
            for mask_path in find_images(mask_dir, recursive=True)
            if osp.basename(mask_path).startswith(self._mask_prefix)
        ]

        for mask_path in masks:
            item_id = osp.splitext(osp.basename(mask_path))[0][len(self._mask_prefix) :]

            image = images.get(item_id)
            if image:
                image = Image(path=image)

            annotations = []
            mask = lazy_mask(mask_path, self._categories[AnnotationType.mask].inverse_colormap)
            mask = mask()  # loading mask through cache

            classes = np.unique(mask)
            for label_id in classes:
                annotations.append(
                    Mask(image=self._lazy_extract_mask(mask, label_id), label=label_id)
                )

            items[item_id] = DatasetItem(
                id=item_id, subset=self._subset, media=image, annotations=annotations
            )

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class CommonSemanticSegmentationImporter(Importer):
    """CommonSemanticSegmentation is introduced in the accuracy checker tool of OpenVINO™
    to cover a general format of datasets for semantic segmentation task.
    This should have the following structure:

    └─ Dataset/
        ├── dataset_meta.json # a list of labels
        ├── images/
        │   ├── <img1>.png
        │   ├── <img2>.png
        │   └── ...
        └── masks/
            ├── <img1>.png
            ├── <img2>.png
            └── ...
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--image-prefix", default="", help="Image prefix (default: '')")
        parser.add_argument("--mask-prefix", default="", help="Mask prefix (default: '')")
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        path = context.require_file(f"**/{DATASET_META_FILE}")
        path = osp.dirname(path)

        context.require_file(osp.join(path, CommonSemanticSegmentationPath.IMAGES_DIR, "**", "*"))
        context.require_file(osp.join(path, CommonSemanticSegmentationPath.MASKS_DIR, "**", "*"))

        return FormatDetectionConfidence.MEDIUM

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "common_semantic_segmentation"}]


@with_subset_dirs
class CommonSemanticSegmentationWithSubsetDirsImporter(CommonSemanticSegmentationImporter):
    """It supports the following subset sub-directory structure for CommonSemanticSegmentation.

    Dataset/
    └─ <split: train,val, ...>
        ├── dataset_meta.json # a list of labels
        ├── images/
        │   ├── <img1>.png
        │   ├── <img2>.png
        │   └── ...
        └── masks/
            ├── <img1>.png
            ├── <img2>.png
            └── ...

    Then, the imported dataset will have train, val, ... CommonSemanticSegmentation subsets.
    """
