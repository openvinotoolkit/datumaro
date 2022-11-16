# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories, Mask
from datumaro.components.extractor import DatasetItem, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import MultiframeImage
from datumaro.util.pickle_util import PickleLoader


class BratsNumpyPath:
    IDS_FILE = "val_ids.p"
    BOXES_FILE = "val_brain_bbox.p"
    LABELS_FILE = "labels"
    DATA_SUFFIX = "_data_cropped"
    LABEL_SUFFIX = "_label_cropped"


class BratsNumpyExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(media_type=MultiframeImage)

        self._root_dir = osp.dirname(path)
        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_path = osp.join(self._root_dir, BratsNumpyPath.LABELS_FILE)
        if osp.isfile(labels_path):
            with open(labels_path, encoding="utf-8") as f:
                for line in f:
                    label_cat.add(line.strip())

        return {AnnotationType.label: label_cat}

    def _load_items(self, path):
        items = {}

        with open(path, "rb") as f:
            ids = PickleLoader.restricted_load(f)

        boxes = None
        boxes_file = osp.join(self._root_dir, BratsNumpyPath.BOXES_FILE)
        if osp.isfile(boxes_file):
            with open(boxes_file, "rb") as f:
                boxes = PickleLoader.restricted_load(f)

        for i, item_id in enumerate(ids):
            image_path = osp.join(self._root_dir, item_id + BratsNumpyPath.DATA_SUFFIX + ".npy")
            media = None
            if osp.isfile(image_path):
                data = np.load(image_path)[0].transpose()
                images = [0] * data.shape[2]
                for j in range(data.shape[2]):
                    images[j] = data[:, :, j]

                media = MultiframeImage(images, path=image_path)

            anno = []
            mask_path = osp.join(self._root_dir, item_id + BratsNumpyPath.LABEL_SUFFIX + ".npy")
            if osp.isfile(mask_path):
                mask = np.load(mask_path)[0].transpose()
                for j in range(mask.shape[2]):
                    classes = np.unique(mask[:, :, j])
                    for class_id in classes:
                        anno.append(
                            Mask(
                                image=self._lazy_extract_mask(mask[:, :, j], class_id),
                                label=class_id,
                                attributes={"image_id": j},
                            )
                        )

            if boxes is not None:
                box = boxes[i]
                anno.append(Cuboid3d(position=list(box[0]), rotation=list(box[1])))

            items[item_id] = DatasetItem(id=item_id, media=media, annotations=anno)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class BratsNumpyImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(BratsNumpyPath.IDS_FILE)

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(
            path, "", "brats_numpy", filename=BratsNumpyPath.IDS_FILE
        )
