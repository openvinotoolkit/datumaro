# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.media import Image
from datumaro.util.image import find_images, load_image

from .format import MvtecPath, MvtecTask


class _MvtecBase(SubsetBase):
    def __init__(self, path, task, subset=None):
        assert osp.isdir(path), path
        self._path = path
        self._task = task

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        self._subset = subset
        super().__init__(subset=subset)

        self._categories = self._load_categories()
        print("base.py 31", self._categories)
        self._items = list(self._load_items().values())
        print("base.py 33", self._items)

    def _load_categories(self):
        label_path = os.listdir(self._path)
        label_cat = LabelCategories()
        for dirname in sorted(set(label_path)):
            label_cat.add(dirname)
        return {AnnotationType.label: label_cat}

    def _load_items(self):
        items = {}
        for image_path in find_images(self._path, recursive=True, max_depth=1):
            label = osp.basename(osp.dirname(image_path))
            label_id = self._categories[AnnotationType.label].find(label)[0]
            item_id = label + "/" + osp.splitext(osp.basename(image_path))[0]

            item = items.get(item_id)
            if item is None:
                item = DatasetItem(id=item_id, subset=self._subset, media=Image(path=image_path))
                items[item_id] = item

            anns = item.annotations
            if self._task == MvtecTask.classification:
                anns.append(Label(label=label_id))
            elif self._task == MvtecTask.segmentation:
                mask_path = osp.join(self._path, MvtecPath.MASK_DIR)
                file_name = item_id + MvtecPath.MASK_POSTFIX + MvtecPath.MASK_EXT
                instance_path = osp.join(mask_path, file_name)

                if osp.exists(instance_path):
                    anns.append(
                        Mask(
                            image=load_image(instance_path, dtype=np.int32),
                            label=label_id,
                        )
                    )
                else:
                    anns.append(Label(label=label_id))
            elif self._task == MvtecTask.detection:
                mask_path = osp.join(self._path, MvtecPath.MASK_DIR)
                file_name = item_id + MvtecPath.MASK_POSTFIX + MvtecPath.MASK_EXT
                instance_path = osp.join(mask_path, file_name)

                if osp.exists(instance_path):
                    instances_mask = load_image(instance_path, dtype=np.int32)

                    from datumaro.util.mask_tools import mask_to_bboxes

                    bboxes = mask_to_bboxes(instances_mask)
                    for bbox in bboxes:
                        anns.append(
                            Bbox(
                                x=bbox[0],
                                y=bbox[2],
                                w=bbox[1] - bbox[0],
                                h=bbox[3] - bbox[2],
                                label=label_id,
                            )
                        )
                else:
                    anns.append(Label(label=label_id))
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class MvtecClassificationBase(_MvtecBase):
    def __init__(self, path):
        super().__init__(path, task=MvtecTask.classification)


class MvtecSegmentationBase(_MvtecBase):
    def __init__(self, path):
        super().__init__(path, task=MvtecTask.segmentation)


class MvtecDetectionBase(_MvtecBase):
    def __init__(self, path):
        super().__init__(path, task=MvtecTask.detection)
