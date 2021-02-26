
# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict

from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import (AnnotationType, DEFAULT_SUBSET_NAME,
    DatasetItem)

from .format import YoloPath


def _make_yolo_bbox(img_size, box):
    # https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
    # <x> <y> <width> <height> - values relative to width and height of image
    # <x> <y> - are center of rectangle
    x = (box[0] + box[2]) / 2 / img_size[0]
    y = (box[1] + box[3]) / 2 / img_size[1]
    w = (box[2] - box[0]) / img_size[0]
    h = (box[3] - box[1]) / img_size[1]
    return x, y, w, h

class YoloConverter(Converter):
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        extractor = self._extractor
        save_dir = self._save_dir

        os.makedirs(save_dir, exist_ok=True)

        label_categories = extractor.categories()[AnnotationType.label]
        label_ids = {label.name: idx
            for idx, label in enumerate(label_categories.items)}
        with open(osp.join(save_dir, 'obj.names'), 'w') as f:
            f.writelines('%s\n' % l[0]
                for l in sorted(label_ids.items(), key=lambda x: x[1]))

        subset_lists = OrderedDict()

        for subset_name, subset in self._extractor.subsets().items():
            if not subset_name or subset_name == DEFAULT_SUBSET_NAME:
                subset_name = YoloPath.DEFAULT_SUBSET_NAME
            elif subset_name not in YoloPath.SUBSET_NAMES:
                log.warn("Skipping subset export '%s'. "
                    "If specified, the only valid names are %s" % \
                    (subset_name, ', '.join(
                        "'%s'" % s for s in YoloPath.SUBSET_NAMES)))
                continue

            subset_dir = osp.join(save_dir, 'obj_%s_data' % subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            image_paths = OrderedDict()

            for item in subset:
                if not item.has_image:
                    raise Exception("Failed to export item '%s': "
                        "item has no image info" % item.id)
                height, width = item.image.size

                image_name = self._make_image_filename(item)
                if self._save_images:
                    if item.has_image and item.image.has_data:
                        self._save_image(item, osp.join(subset_dir, image_name))
                    else:
                        log.warning("Item '%s' has no image" % item.id)
                image_paths[item.id] = osp.join('data',
                    osp.basename(subset_dir), image_name)

                yolo_annotation = ''
                for bbox in item.annotations:
                    if bbox.type is not AnnotationType.bbox:
                        continue
                    if bbox.label is None:
                        continue

                    yolo_bb = _make_yolo_bbox((width, height), bbox.points)
                    yolo_bb = ' '.join('%.6f' % p for p in yolo_bb)
                    yolo_annotation += '%s %s\n' % (bbox.label, yolo_bb)

                annotation_path = osp.join(subset_dir, '%s.txt' % item.id)
                os.makedirs(osp.dirname(annotation_path), exist_ok=True)
                with open(annotation_path, 'w') as f:
                    f.write(yolo_annotation)

            subset_list_name = '%s.txt' % subset_name
            subset_lists[subset_name] = subset_list_name
            with open(osp.join(save_dir, subset_list_name), 'w') as f:
                f.writelines('%s\n' % s for s in image_paths.values())

        with open(osp.join(save_dir, 'obj.data'), 'w') as f:
            f.write('classes = %s\n' % len(label_ids))

            for subset_name, subset_list_name in subset_lists.items():
                f.write('%s = %s\n' % (subset_name,
                    osp.join('data', subset_list_name)))

            f.write('names = %s\n' % osp.join('data', 'obj.names'))
            f.write('backup = backup/\n')

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            cls.convert(dataset.get_subset(subset), save_dir=save_dir, **kwargs)

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.has_image):
                continue

            if subset == DEFAULT_SUBSET_NAME:
                subset = YoloPath.DEFAULT_SUBSET_NAME
            image_path = osp.join(save_dir, 'obj_%s_data' % subset,
                conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)