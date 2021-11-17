# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp

from defusedxml import ElementTree
import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Bbox, CompiledMask, Label, Mask,
)
from datumaro.components.extractor import DatasetItem, SourceExtractor
from datumaro.components.media import Image
from datumaro.util.image import find_images
from datumaro.util.mask_tools import invert_colormap, lazy_mask
from datumaro.util.meta_file_util import is_meta_file, parse_meta_file

from .format import VocInstColormap, VocPath, VocTask, make_voc_categories

_inverse_inst_colormap = invert_colormap(VocInstColormap)

class _VocExtractor(SourceExtractor):
    def __init__(self, path, task):
        assert osp.isfile(path), path
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(osp.dirname(path)))
        self._task = task

        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        self._categories = self._load_categories(self._dataset_dir)

        label_color = lambda label_idx: \
            self._categories[AnnotationType.mask].colormap.get(label_idx, None)
        log.debug("Loaded labels: %s" % ', '.join(
            "'%s' %s" % (l.name, ('(%s, %s, %s)' % c) if c else '')
            for i, l, c in ((i, l, label_color(i)) for i, l in enumerate(
                self._categories[AnnotationType.label].items
            ))
        ))
        self._items = { item: None for item in self._load_subset_list(path) }

    def _get_label_id(self, label):
        label_id, _ = self._categories[AnnotationType.label].find(label)
        assert label_id is not None, label
        return label_id

    def _load_categories(self, dataset_path):
        label_map = None
        if (is_meta_file(dataset_path)):
            label_map = parse_meta_file(dataset_path)

        return make_voc_categories(label_map)

    def _load_subset_list(self, subset_path):
        subset_list = []
        with open(subset_path, encoding='utf-8') as f:
            for line in f:
                if self._task == VocTask.person_layout:
                    objects = line.split('\"')
                    if 1 < len(objects):
                        if len(objects) == 3:
                            line = objects[1]
                        else:
                            raise Exception("Line %s: unexpected number "
                                "of quotes in filename" % line)
                    else:
                        line = line.split()[0]
                else:
                    line = line.strip()
                subset_list.append(line)
            return subset_list

class VocClassificationExtractor(_VocExtractor):
    def __init__(self, path):
        super().__init__(path, VocTask.classification)

    def __iter__(self):
        annotations = self._load_annotations()

        image_dir = osp.join(self._dataset_dir, VocPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for item_id in self._items:
            log.debug("Reading item '%s'" % item_id)
            yield DatasetItem(id=item_id, subset=self._subset,
                image=images.get(item_id), annotations=annotations.get(item_id))

    def _load_annotations(self):
        annotations = {}
        task_dir = osp.dirname(self._path)
        for label_id, label in enumerate(self._categories[AnnotationType.label]):
            ann_file = osp.join(task_dir, f'{label.name}_{self._subset}.txt')
            if not osp.isfile(ann_file):
                continue

            with open(ann_file, encoding='utf-8') as f:
                for line in f:
                    item, present = line.rsplit(maxsplit=1)
                    if present == '1':
                        annotations.setdefault(item, []).append(Label(label_id))

        return annotations

class _VocXmlExtractor(_VocExtractor):
    def __init__(self, path, task):
        super().__init__(path, task)

    def __iter__(self):
        anno_dir = osp.join(self._dataset_dir, VocPath.ANNOTATIONS_DIR)

        for item_id in self._items:
            log.debug("Reading item '%s'" % item_id)
            image = item_id + VocPath.IMAGE_EXT
            height, width = 0, 0

            anns = []
            ann_file = osp.join(anno_dir, item_id + '.xml')
            if osp.isfile(ann_file):
                root_elem = ElementTree.parse(ann_file)
                height = root_elem.find('size/height')
                if height is not None:
                    height = int(height.text)
                width = root_elem.find('size/width')
                if width is not None:
                    width = int(width.text)
                filename_elem = root_elem.find('filename')
                if filename_elem is not None:
                    image = filename_elem.text
                anns = self._parse_annotations(root_elem)

            image = osp.join(self._dataset_dir, VocPath.IMAGES_DIR, image)
            if height and width:
                image = Image(path=image, size=(height, width))

            yield DatasetItem(id=item_id, subset=self._subset,
                image=image, annotations=anns)

    def _parse_annotations(self, root_elem):
        item_annotations = []

        for obj_id, object_elem in enumerate(root_elem.findall('object')):
            obj_id += 1
            attributes = {}
            group = obj_id

            obj_label_id = None
            label_elem = object_elem.find('name')
            if label_elem is not None:
                obj_label_id = self._get_label_id(label_elem.text)

            obj_bbox = self._parse_bbox(object_elem)

            if obj_label_id is None or obj_bbox is None:
                continue

            difficult_elem = object_elem.find('difficult')
            attributes['difficult'] = difficult_elem is not None and \
                difficult_elem.text == '1'

            truncated_elem = object_elem.find('truncated')
            attributes['truncated'] = truncated_elem is not None and \
                truncated_elem.text == '1'

            occluded_elem = object_elem.find('occluded')
            attributes['occluded'] = occluded_elem is not None and \
                occluded_elem.text == '1'

            pose_elem = object_elem.find('pose')
            if pose_elem is not None:
                attributes['pose'] = pose_elem.text

            point_elem = object_elem.find('point')
            if point_elem is not None:
                point_x = point_elem.find('x')
                point_y = point_elem.find('y')
                point = [float(point_x.text), float(point_y.text)]
                attributes['point'] = point

            actions_elem = object_elem.find('actions')
            actions = {a: False
                for a in self._categories[AnnotationType.label] \
                    .items[obj_label_id].attributes}
            if actions_elem is not None:
                for action_elem in actions_elem:
                    actions[action_elem.tag] = (action_elem.text == '1')
            for action, present in actions.items():
                attributes[action] = present

            has_parts = False
            for part_elem in object_elem.findall('part'):
                part = part_elem.find('name').text
                part_label_id = self._get_label_id(part)
                part_bbox = self._parse_bbox(part_elem)

                if self._task is not VocTask.person_layout:
                    break
                if part_bbox is None:
                    continue
                has_parts = True
                item_annotations.append(Bbox(*part_bbox, label=part_label_id,
                    group=group))

            attributes_elem = object_elem.find('attributes')
            if attributes_elem is not None:
                for attr_elem in attributes_elem.iter('attribute'):
                    attributes[attr_elem.find('name').text] = \
                        attr_elem.find('value').text

            if self._task is VocTask.person_layout and not has_parts:
                continue
            if self._task is VocTask.action_classification and not actions:
                continue

            item_annotations.append(Bbox(*obj_bbox, label=obj_label_id,
                attributes=attributes, id=obj_id, group=group))

        return item_annotations

    @staticmethod
    def _parse_bbox(object_elem):
        bbox_elem = object_elem.find('bndbox')
        xmin = float(bbox_elem.find('xmin').text)
        xmax = float(bbox_elem.find('xmax').text)
        ymin = float(bbox_elem.find('ymin').text)
        ymax = float(bbox_elem.find('ymax').text)
        return [xmin, ymin, xmax - xmin, ymax - ymin]

class VocDetectionExtractor(_VocXmlExtractor):
    def __init__(self, path):
        super().__init__(path, task=VocTask.detection)

class VocLayoutExtractor(_VocXmlExtractor):
    def __init__(self, path):
        super().__init__(path, task=VocTask.person_layout)

class VocActionExtractor(_VocXmlExtractor):
    def __init__(self, path):
        super().__init__(path, task=VocTask.action_classification)

class VocSegmentationExtractor(_VocExtractor):
    def __init__(self, path):
        super().__init__(path, task=VocTask.segmentation)

    def __iter__(self):
        image_dir = osp.join(self._dataset_dir, VocPath.IMAGES_DIR)
        if osp.isdir(image_dir):
            images = {
                osp.splitext(osp.relpath(p, image_dir))[0].replace('\\', '/'): p
                for p in find_images(image_dir, recursive=True)
            }
        else:
            images = {}

        for item_id in self._items:
            log.debug("Reading item '%s'" % item_id)
            anns = self._load_annotations(item_id)
            yield DatasetItem(id=item_id, subset=self._subset,
                image=images.get(item_id), annotations=anns)

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

    def _load_annotations(self, item_id):
        item_annotations = []

        class_mask = None
        segm_path = osp.join(self._dataset_dir, VocPath.SEGMENTATION_DIR,
            item_id + VocPath.SEGM_EXT)
        if osp.isfile(segm_path):
            inverse_cls_colormap = \
                self._categories[AnnotationType.mask].inverse_colormap
            class_mask = lazy_mask(segm_path, inverse_cls_colormap)

        instances_mask = None
        inst_path = osp.join(self._dataset_dir, VocPath.INSTANCES_DIR,
            item_id + VocPath.SEGM_EXT)
        if osp.isfile(inst_path):
            instances_mask = lazy_mask(inst_path, _inverse_inst_colormap)

        if instances_mask is not None:
            compiled_mask = CompiledMask(class_mask, instances_mask)

            label_cat = self._categories[AnnotationType.label]

            if class_mask is not None:
                instance_labels = compiled_mask.get_instance_labels()
            else:
                instance_labels = {i: None
                    for i in range(compiled_mask.instance_count)}

            for instance_id, label_id in instance_labels.items():
                if len(label_cat) <= label_id:
                    raise Exception(
                        "Item %s: a mask has unexpected class number %s" %
                        (item_id, label_id))

                image = compiled_mask.lazy_extract(instance_id)

                item_annotations.append(Mask(image=image, label=label_id,
                    group=instance_id))
        elif class_mask is not None:
            log.warning("Item %s: only class segmentations available" % item_id)

            class_mask = class_mask()
            classes = np.unique(class_mask)
            for label_id in classes:
                image = self._lazy_extract_mask(class_mask, label_id)
                item_annotations.append(Mask(image=image, label=label_id))

        return item_annotations
