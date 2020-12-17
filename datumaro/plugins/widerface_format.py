
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import re

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Importer, SourceExtractor)


class WiderFacePath:
    IMAGE_EXT = '.jpg'
    ANNOTATIONS_DIR = 'wider_face_split'
    IMAGES_DIR = 'images'
    SUBSET_DIR = 'WIDER_'
    BBOX_ATTRIBUTES = ['blur', 'expression', 'illumination',
        'occlusion', 'pose', 'invalid']

class WiderFaceExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise Exception("Can't read annotation file '%s'" % path)
        self._path = path
        self._dataset_dir = osp.dirname(osp.dirname(path))

        subset = osp.splitext(osp.basename(path))[0]
        match = re.fullmatch(r'wider_face_\S+_bbx_gt', subset)
        if match:
            subset = subset.split('_')[2]
        super().__init__(subset=subset)

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}
        with open(path, 'r') as f:
            lines = f.readlines()

        image_ids = [image_id for image_id, line in enumerate(lines)
            if WiderFacePath.IMAGE_EXT in line]

        for image_id in image_ids:
            image = lines[image_id]
            image_path = osp.join(self._dataset_dir, WiderFacePath.SUBSET_DIR
                + self._subset, WiderFacePath.IMAGES_DIR, image[:-1])
            item_id = image[:-(len(WiderFacePath.IMAGE_EXT) + 1)]

            bbox_count = lines[image_id + 1]
            bbox_lines = lines[image_id + 2 : image_id + int(bbox_count) + 2]
            annotations = []
            for bbox in bbox_lines:
                bbox_list = bbox.split()
                if len(bbox_list) >= 4:
                    attributes = {}
                    if len(bbox_list) == 10:
                        i = 4
                        for attr in WiderFacePath.BBOX_ATTRIBUTES:
                            if bbox_list[i] != '-':
                                attributes[attr] = int(bbox_list[i])
                            i += 1
                    annotations.append(Bbox(
                        int(bbox_list[0]), int(bbox_list[1]),
                        int(bbox_list[2]), int(bbox_list[3]),
                        attributes = attributes
                    ))

            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=image_path, annotations=annotations)
        return items

class WiderFaceImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(osp.join(path,
            WiderFacePath.ANNOTATIONS_DIR), '.txt', 'wider_face')

class WiderFaceConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    def apply(self):
        save_dir = self._save_dir

        os.makedirs(save_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            subset_dir = osp.join(save_dir, WiderFacePath.SUBSET_DIR + subset_name)

            wider_annotation = ''
            for item in subset:
                wider_annotation += '%s\n' % (item.id + WiderFacePath.IMAGE_EXT)
                if item.has_image and self._save_images:
                    self._save_image(item, osp.join(save_dir, subset_dir,
                        WiderFacePath.IMAGES_DIR, item.id + WiderFacePath.IMAGE_EXT))

                bboxes = [a for a in item.annotations
                    if a.type == AnnotationType.bbox]

                wider_annotation += '%s\n' % len(bboxes)
                for bbox in bboxes:
                    wider_bb = ' '.join('%d' % p for p in bbox.get_bbox())
                    wider_annotation += '%s ' % wider_bb
                    if bbox.attributes:
                        wider_attr = ''
                        attr_counter = 0
                        for attr in WiderFacePath.BBOX_ATTRIBUTES:
                            if attr in bbox.attributes:
                                wider_attr += '%s ' % bbox.attributes[attr]
                                attr_counter += 1
                            else:
                                wider_attr += '- '
                        if attr_counter > 0:
                            wider_annotation += wider_attr
                    wider_annotation  += '\n'
            annotation_path = osp.join(save_dir, WiderFacePath.ANNOTATIONS_DIR,
                'wider_face_' + subset_name + '_bbx_gt.txt')
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)
            with open(annotation_path, 'w') as f:
                f.write(wider_annotation)
