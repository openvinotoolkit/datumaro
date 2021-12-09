# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

# pylint: disable=no-self-use

import json
import os
import os.path as osp
import shutil

import numpy as np
import pycocotools.mask as mask_utils

from datumaro.components.annotation import (
    Annotation, Bbox, Caption, Cuboid3d, Label, LabelCategories, Mask,
    MaskCategories, Points, PointsCategories, Polygon, PolyLine, RleMask,
    _Shape,
)
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem
from datumaro.util import cast

from .format import DatumaroPath


class _SubsetWriter:
    def __init__(self, context):
        self._context = context

        self._data = {
            'info': {},
            'categories': {},
            'items': [],
        }

    @property
    def categories(self):
        return self._data['categories']

    @property
    def items(self):
        return self._data['items']

    def is_empty(self):
        return not self.items

    def add_item(self, item):
        annotations = []
        item_desc = {
            'id': item.id,
            'annotations': annotations,
        }

        if item.attributes:
            item_desc['attr'] = item.attributes

        if item.has_image:
            path = item.image.path
            if self._context._save_images:
                path = self._context._make_image_filename(item)
                self._context._save_image(item,
                    osp.join(self._context._images_dir, item.subset, path))

            item_desc['image'] = {
                'path': path,
            }
            if item.image.has_size: # avoid occasional loading
                item_desc['image']['size'] = item.image.size

        if item.has_point_cloud:
            path = item.point_cloud
            if self._context._save_images:
                path = self._context._make_pcd_filename(item)
                self._context._save_point_cloud(item,
                    osp.join(self._context._pcd_dir, item.subset, path))

            item_desc['point_cloud'] = {
                'path': path
            }

        if item.related_images:
            images = sorted(item.related_images, key=lambda i: i.path)
            if self._context._save_images:
                related_images = []
                for i, img in enumerate(images):
                    ri_desc = {}

                    # Images can have completely the same names or don't have
                    # them at all, so we just rename them
                    ri_desc['path'] = \
                        f'image_{i}{self._context._find_image_ext(img)}'

                    if img.has_data:
                        img.save(osp.join(self._context._related_images_dir,
                            item.subset, item.id, ri_desc['path']))
                    if img.has_size:
                        ri_desc['size'] = img.size
                    related_images.append(ri_desc)
            else:
                related_images = [{'path': img.path} for img in images]

            item_desc['related_images'] = related_images

        self.items.append(item_desc)

        for ann in item.annotations:
            if isinstance(ann, Label):
                converted_ann = self._convert_label_object(ann)
            elif isinstance(ann, Mask):
                converted_ann = self._convert_mask_object(ann)
            elif isinstance(ann, Points):
                converted_ann = self._convert_points_object(ann)
            elif isinstance(ann, PolyLine):
                converted_ann = self._convert_polyline_object(ann)
            elif isinstance(ann, Polygon):
                converted_ann = self._convert_polygon_object(ann)
            elif isinstance(ann, Bbox):
                converted_ann = self._convert_bbox_object(ann)
            elif isinstance(ann, Caption):
                converted_ann = self._convert_caption_object(ann)
            elif isinstance(ann, Cuboid3d):
                converted_ann = self._convert_cuboid_3d_object(ann)
            else:
                raise NotImplementedError()
            annotations.append(converted_ann)

    def add_categories(self, categories):
        for ann_type, desc in categories.items():
            if isinstance(desc, LabelCategories):
                converted_desc = self._convert_label_categories(desc)
            elif isinstance(desc, MaskCategories):
                converted_desc = self._convert_mask_categories(desc)
            elif isinstance(desc, PointsCategories):
                converted_desc = self._convert_points_categories(desc)
            else:
                raise NotImplementedError()
            self.categories[ann_type.name] = converted_desc

    def write(self, ann_file):
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False)

    def _convert_annotation(self, obj):
        assert isinstance(obj, Annotation)

        ann_json = {
            'id': cast(obj.id, int),
            'type': cast(obj.type.name, str),
            'attributes': obj.attributes,
            'group': cast(obj.group, int, 0),
        }
        return ann_json

    def _convert_label_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update({
            'label_id': cast(obj.label, int),
        })
        return converted

    def _convert_mask_object(self, obj):
        converted = self._convert_annotation(obj)

        if isinstance(obj, RleMask):
            rle = obj.rle
        else:
            rle = mask_utils.encode(
                np.require(obj.image, dtype=np.uint8, requirements='F'))

        if isinstance(rle['counts'], str):
            counts = rle['counts']
        else:
            counts = rle['counts'].decode('ascii')

        converted.update({
            'label_id': cast(obj.label, int),
            'rle': {
                # serialize as compressed COCO mask
                'counts': counts,
                'size': list(int(c) for c in rle['size']),
            },
            'z_order': obj.z_order,
        })
        return converted

    def _convert_shape_object(self, obj):
        assert isinstance(obj, _Shape)
        converted = self._convert_annotation(obj)

        converted.update({
            'label_id': cast(obj.label, int),
            'points': [float(p) for p in obj.points],
            'z_order': obj.z_order,
        })
        return converted

    def _convert_polyline_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_polygon_object(self, obj):
        return self._convert_shape_object(obj)

    def _convert_bbox_object(self, obj):
        converted = self._convert_shape_object(obj)
        converted.pop('points', None)
        converted['bbox'] = [float(p) for p in obj.get_bbox()]
        return converted

    def _convert_points_object(self, obj):
        converted = self._convert_shape_object(obj)

        converted.update({
            'visibility': [int(v.value) for v in obj.visibility],
        })
        return converted

    def _convert_caption_object(self, obj):
        converted = self._convert_annotation(obj)

        converted.update({
            'caption': cast(obj.caption, str),
        })
        return converted

    def _convert_cuboid_3d_object(self, obj):
        converted = self._convert_annotation(obj)
        converted.update({
            'label_id': cast(obj.label, int),
            'position': [float(p) for p in obj.position],
            'rotation': [float(p) for p in obj.rotation],
            'scale': [float(p) for p in obj.scale]
        })
        return converted

    def _convert_attribute_categories(self, attributes):
        return sorted(attributes)

    def _convert_label_categories(self, obj):
        converted = {
            'labels': [],
            'attributes': self._convert_attribute_categories(obj.attributes),
        }
        for label in obj.items:
            converted['labels'].append({
                'name': cast(label.name, str),
                'parent': cast(label.parent, str),
                'attributes': self._convert_attribute_categories(label.attributes),
            })
        return converted

    def _convert_mask_categories(self, obj):
        converted = {
            'colormap': [],
        }
        for label_id, color in obj.colormap.items():
            converted['colormap'].append({
                'label_id': int(label_id),
                'r': int(color[0]),
                'g': int(color[1]),
                'b': int(color[2]),
            })
        return converted

    def _convert_points_categories(self, obj):
        converted = {
            'items': [],
        }
        for label_id, item in obj.items.items():
            converted['items'].append({
                'label_id': int(label_id),
                'labels': [cast(label, str) for label in item.labels],
                'joints': [list(map(int, j)) for j in item.joints],
            })
        return converted

class DatumaroConverter(Converter):
    DEFAULT_IMAGE_EXT = DatumaroPath.IMAGE_EXT

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        images_dir = osp.join(self._save_dir, DatumaroPath.IMAGES_DIR)
        os.makedirs(images_dir, exist_ok=True)
        self._images_dir = images_dir

        annotations_dir = osp.join(self._save_dir, DatumaroPath.ANNOTATIONS_DIR)
        os.makedirs(annotations_dir, exist_ok=True)
        self._annotations_dir = annotations_dir

        self._pcd_dir = osp.join(self._save_dir, DatumaroPath.PCD_DIR)
        self._related_images_dir = osp.join(self._save_dir,
            DatumaroPath.RELATED_IMAGES_DIR)

        writers = {s: _SubsetWriter(self) for s in self._extractor.subsets()}
        for writer in writers.values():
            writer.add_categories(self._extractor.categories())

        for item in self._extractor:
            subset = item.subset or DEFAULT_SUBSET_NAME
            writers[subset].add_item(item)

        for subset, writer in writers.items():
            ann_file = osp.join(self._annotations_dir, '%s.json' % subset)

            if self._patch and subset in self._patch.updated_subsets and \
                    writer.is_empty():
                if osp.isfile(ann_file):
                    # Remove subsets that became empty
                    os.remove(ann_file)
                continue

            writer.write(ann_file)

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        for subset in patch.updated_subsets:
            conv = cls(dataset.get_subset(subset), save_dir=save_dir, **kwargs)
            conv._patch = patch
            conv.apply()

        conv = cls(dataset, save_dir=save_dir, **kwargs)
        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or \
                    not item.has_image and not item.has_point_cloud):
                continue

            image_path = osp.join(save_dir, DatumaroPath.IMAGES_DIR,
                item.subset, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.unlink(image_path)

            pcd_path = osp.join(save_dir, DatumaroPath.PCD_DIR,
                item.subset, conv._make_pcd_filename(item))
            if osp.isfile(pcd_path):
                os.unlink(pcd_path)

            related_images_path = osp.join(save_dir,
                DatumaroPath.RELATED_IMAGES_DIR, item.subset, item.id)
            if osp.isdir(related_images_path):
                shutil.rmtree(related_images_path)
