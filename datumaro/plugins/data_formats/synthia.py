# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.image import find_images, load_image
from datumaro.util.mask_tools import generate_colormap, lazy_mask
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class SynthiaPath:
    IMAGES_DIR = "RGB"
    LABELS_SEGM_DIR = "GT/LABELS"
    SEMANTIC_SEGM_DIR = "GT/COLOR"
    LABELMAP_FILE = "label_colors.txt"


SYNTHIA_LABEL_MAP = OrderedDict(
    [
        ("Void", (0, 0, 0)),
        ("Sky", (128, 128, 128)),
        ("Building", (128, 0, 0)),
        ("Road", (128, 64, 128)),
        ("Sidewalk", (0, 0, 192)),
        ("Fence", (64, 64, 128)),
        ("Vegetation", (128, 128, 0)),
        ("Pole", (192, 192, 128)),
        ("Car", (64, 0, 128)),
        ("TrafficSign", (192, 128, 128)),
        ("Pedestrian", (64, 64, 0)),
        ("Bicycle", (0, 128, 192)),
        ("Lanemarking", (0, 172, 0)),
        ("Reserved_1", (0, 0, 0)),
        ("Reserved_2", (0, 0, 0)),
        ("TrafficLight", (0, 128, 128)),
    ]
)


def make_categories(label_map=None):
    if label_map is None:
        label_map = SYNTHIA_LABEL_MAP

    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors:  # generate new colors
        colormap = generate_colormap(len(label_map))
    else:  # only copy defined colors
        colormap = {
            label_id: (desc[0], desc[1], desc[2])
            for label_id, desc in enumerate(label_map.values())
        }
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap  # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories
    return categories


def parse_label_map(path):
    label_map = OrderedDict()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # skip empty and commented lines
            line = line.strip()
            if not line or line[0] == "#":
                continue

            # color, name
            label_desc = line.split()

            if 2 < len(label_desc):
                name = label_desc[3]
                color = tuple([int(c) for c in label_desc[:3]])
            else:
                name = label_desc[0]
                color = None

            if name in label_map:
                raise ValueError("Label '%s' is already defined" % name)

            label_map[name] = color
    return label_map


class SynthiaBase(SubsetBase):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__()

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        if has_meta_file(path):
            return make_categories(parse_meta_file(path))
        label_map_path = osp.join(path, SynthiaPath.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = SYNTHIA_LABEL_MAP
        return make_categories(label_map)

    def _load_items(self, root_dir):
        image_dir = osp.join(root_dir, SynthiaPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        items = {}

        inst_dir = osp.join(root_dir, SynthiaPath.LABELS_SEGM_DIR)
        if osp.isdir(inst_dir):
            gt_images = find_images(inst_dir, recursive=True)
            for gt_img in gt_images:
                item_id = osp.splitext(osp.relpath(gt_img, inst_dir))[0].replace("\\", "/")

                anno = []
                labels_mask = load_image(gt_img, dtype=np.uint16)
                dynamic_objects = np.unique(labels_mask[:, :, 1])
                labels_mask = labels_mask[:, :, 2]
                segm_ids = np.unique(labels_mask)
                for segm_id in segm_ids:
                    attr = {"dynamic_object": False}
                    if segm_id != 0 and segm_id in dynamic_objects:
                        attr["dynamic_object"] = True
                    anno.append(
                        Mask(
                            image=self._lazy_extract_mask(labels_mask, segm_id),
                            label=segm_id,
                            attributes=attr,
                        )
                    )

                image = images.get(item_id)
                if image:
                    image = Image(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)

        elif osp.isdir(osp.join(root_dir, SynthiaPath.SEMANTIC_SEGM_DIR)):
            gt_dir = osp.join(root_dir, SynthiaPath.SEMANTIC_SEGM_DIR)
            gt_images = find_images(gt_dir, recursive=True)
            for gt_img in gt_images:
                item_id = osp.splitext(osp.relpath(gt_img, gt_dir))[0].replace("\\", "/")

                anno = []
                inverse_cls_colormap = self._categories[AnnotationType.mask].inverse_colormap
                color_mask = lazy_mask(gt_img, inverse_cls_colormap)
                color_mask = color_mask()
                classes = np.unique(color_mask)
                for label_id in classes:
                    anno.append(
                        Mask(image=self._lazy_extract_mask(color_mask, label_id), label=label_id)
                    )

                image = images.get(item_id)
                if image:
                    image = Image(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class SynthiaImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any():
            for prefix in (
                SynthiaPath.IMAGES_DIR,
                SynthiaPath.LABELS_SEGM_DIR,
                SynthiaPath.SEMANTIC_SEGM_DIR,
            ):
                with context.alternative():
                    context.require_file(f"{prefix}/**/*.png")

    @classmethod
    def find_sources(cls, path):
        return [{"url": path, "format": "synthia"}]
