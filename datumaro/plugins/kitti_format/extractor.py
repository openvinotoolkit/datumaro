
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import os.path as osp
from glob import iglob

from datumaro.components.extractor import (SourceExtractor,
    AnnotationType, DatasetItem, Mask, Bbox
)
from datumaro.util.image import load_image

from .format import (
    KittiTask, KittiPath, KittiLabelMap, parse_label_map,
    make_kitti_categories, make_kitti_detection_categories
)

class _KittiExtractor(SourceExtractor):
    def __init__(self, path, task, subset=None):
        assert osp.isdir(path), path
        self._path = path
        self._task = task

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        self._subset = subset
        super().__init__(subset=subset)

        self._categories = self._load_categories(osp.join(self._path, '../'))
        self._items = list(self._load_items().values())

    def _load_categories(self, path):
        if self._task == KittiTask.segmentation:
            return self._load_categories_segmentation(path)
        elif self._task == KittiTask.detection:
            return make_kitti_detection_categories()

    def _load_categories_segmentation(self, path):
        label_map = None
        label_map_path = osp.join(path, KittiPath.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = KittiLabelMap
        self._labels = [label for label in label_map]
        return make_kitti_categories(label_map)

    def _load_items(self):
        items = {}

        for image_path in iglob(
                osp.join(self._path, KittiPath.IMAGES_DIR, '**',
                '*%s' % KittiPath.IMAGE_EXT),
                recursive=True):
            image_name = osp.relpath(image_path, osp.join(self._path,
                KittiPath.IMAGES_DIR))
            sample_id = image_name.replace(KittiPath.IMAGE_EXT, '')
            anns = []

            instances_path = osp.join(self._path, KittiPath.INSTANCES_DIR,
                image_name)
            if self._task == KittiTask.segmentation and \
                osp.isfile(instances_path):
                instances_mask = load_image(instances_path, dtype=np.int32)
                segm_ids = np.unique(instances_mask)
                for segm_id in segm_ids:
                    semantic_id = segm_id >> 8
                    ann_id = segm_id % 256
                    isCrowd = bool(ann_id == 0)
                    anns.append(Mask(
                        image=self._lazy_extract_mask(instances_mask, segm_id),
                        label=semantic_id, id=ann_id,
                        attributes = { 'is_crowd': isCrowd }))

            labels_path = osp.join(self._path, KittiPath.LABELS_DIR,
                sample_id+'.txt')
            if self._task == KittiTask.detection and osp.isfile(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line_idx, line in enumerate(lines):
                    attributes = {}
                    line = line.split()
                    assert len(line) == 15
                    x1, y1 = float(line[4]), float(line[5])
                    x2, y2 = float(line[6]), float(line[7])
                    attributes['truncated'] = float(line[1]) != 0
                    attributes['occluded']  = int(line[2]) != 0
                    label_id = self.categories()[
                        AnnotationType.label].find(line[0])[0]
                    anns.append(
                        Bbox(x=x1, y=y1, w=x2-x1, h=y2-y1, id=line_idx,
                            attributes=attributes, label=label_id,
                        ))
            items[sample_id] = DatasetItem(id=sample_id, subset=self._subset,
                image=image_path, annotations=anns)
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

class KittiSegmentationExtractor(_KittiExtractor):
    def __init__(self, path):
        super().__init__(path, task=KittiTask.segmentation)

class KittiDetectionExtractor(_KittiExtractor):
    def __init__(self, path):
        super().__init__(path, task=KittiTask.detection)