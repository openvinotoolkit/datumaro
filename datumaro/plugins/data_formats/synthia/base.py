# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
import os.path as osp

from glob import glob
import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.media import Image
from datumaro.util.image import find_images, load_image
from datumaro.util.mask_tools import generate_colormap
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file

from .format import (
    SynthiaRandPath,
    SynthiaRandLabelMap,
    SynthiaSfPath,
    SynthiaSfLabelMap,
    SynthiaAlPath,
    SynthiaAlLabelMap,
)


class _SynthiaBase(SubsetBase):
    def __init__(self, path):
        super().__init__()

        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        self._path_format = SynthiaRandPath
        self._label_map = SynthiaRandLabelMap

        self._inst_dir = osp.join(path, self._path_format.LABELS_SEGM_DIR)
        self._seg_dir = osp.join(path, self._path_format.SEMANTIC_SEGM_DIR)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())

    def _make_categories(self, label_map=None):
        if label_map is None:
            label_map = self._label_map

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

    def _load_categories(self, path):
        if has_meta_file(path):
            return self._make_categories(parse_meta_file(path))
        label_map_path = osp.join(path, self._path_format.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = self._parse_label_map(label_map_path)
        else:
            label_map = self._label_map
        return self._make_categories(label_map)

    def _load_items(self, root_dir):
        image_dir = osp.join(root_dir, self._path_format.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace("\\", "/"): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        items = {}

        if self._inst_dir:
            gt_labels = glob(self._inst_dir + "/*.txt")
            for gt_label in gt_labels:
                item_id = osp.splitext(osp.relpath(gt_label, self._inst_dir))[0].replace("\\", "/")

                anno = []
                labels_mask = np.loadtxt(gt_label)
                classes = np.unique(labels_mask)
                for label_id in classes:
                    anno.append(
                        Mask(image=self._lazy_extract_mask(labels_mask, label_id), label=label_id)
                    )

                image = images.get(item_id)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)

        else:
            for seg_img_path in find_images(self._seg_dir, recursive=True):
                item_id = osp.splitext(osp.relpath(seg_img_path, self._seg_dir))[0].replace(
                    "\\", "/"
                )

                inverse_cls_colormap = self._categories[AnnotationType.mask].inverse_colormap

                seg_img = load_image(seg_img_path, dtype=np.uint16)
                color_mask = np.zeros_like(seg_img[:, :, 0])
                for i in range(seg_img.shape[0]):
                    for j in range(seg_img.shape[1]):
                        pixel_color = tuple(seg_img[i, j, :])
                        if pixel_color in inverse_cls_colormap.keys():
                            color_mask[i, j] = inverse_cls_colormap[pixel_color]

                classes = np.unique(color_mask)

                anno = []
                for label_id in classes:
                    anno.append(
                        Mask(image=self._lazy_extract_mask(color_mask, label_id), label=label_id)
                    )

                image = images.get(item_id)
                if image:
                    image = Image.from_file(path=image)

                items[item_id] = DatasetItem(id=item_id, media=image, annotations=anno)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

    @staticmethod
    def _parse_label_map(path):
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


class SynthiaRandBase(_SynthiaBase):
    def __init__(self, path):
        super().__init__(path=path)


class SynthiaSfBase(_SynthiaBase):
    def __init__(self, path):
        SubsetBase.__init__(self)

        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        self._label_map = SynthiaSfLabelMap
        self._path_format = SynthiaSfPath

        self._inst_dir = None
        self._seg_dir = osp.join(path, self._path_format.SEMANTIC_SEGM_DIR)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())


class SynthiaAlBase(_SynthiaBase):
    def __init__(self, path):
        SubsetBase.__init__(self)

        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        self._label_map = SynthiaAlLabelMap
        self._path_format = SynthiaAlPath

        self._inst_dir = None
        self._seg_dir = osp.join(path, self._path_format.SEMANTIC_SEGM_DIR)

        self._categories = self._load_categories(path)
        self._items = list(self._load_items(path).values())
