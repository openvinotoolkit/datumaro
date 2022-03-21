# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import glob
import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import generate_colormap, lazy_mask
from datumaro.util.meta_file_util import (
    DATASET_META_FILE,
    has_meta_file,
    is_meta_file,
    parse_meta_file,
)


class CommonSegmentationPath:
    MASKS_DIR = "masks"
    IMAGES_DIR = "images"
    IMAGE_EXT = ".png"


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


class CommonSegmentationExtractor(SourceExtractor):
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

        if has_meta_file(path):
            label_map = parse_meta_file(path)
            self._categories = make_categories(label_map)
        else:
            meta_file = glob.glob(osp.join(path, "**", DATASET_META_FILE), recursive=True)
            if is_meta_file(meta_file[0]):
                label_map = parse_meta_file(meta_file[0])
                self._categories = make_categories(label_map)

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        image_dir = osp.join(path, "**", CommonSegmentationPath.IMAGES_DIR)

        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/")[
                    len(self._image_prefix) :
                ]: p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for mask_path in glob.glob(
            osp.join(path, "**", CommonSegmentationPath.MASKS_DIR, f"{self._mask_prefix}*.*"),
            recursive=True,
        ):
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

            items[item_id] = DatasetItem(id=item_id, media=image, annotations=annotations)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class CommonSegmentationImporter(Importer):
    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("--image-prefix", default="", help="Image prefix (default: '')")
        parser.add_argument("--mask-prefix", default="", help="Mask prefix (default: '')")
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"**/{CommonSegmentationPath.IMAGES_DIR}/**/*.*")
        context.require_file(f"**/{CommonSegmentationPath.MASKS_DIR}/**/*.*")
        context.require_file(f"**/{DATASET_META_FILE}")

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "common_segmentation"}]
