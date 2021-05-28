
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from collections import OrderedDict

from datumaro.components.extractor import (SourceExtractor, DatasetItem,
                                           AnnotationType, Cuboid3D,
                                           LabelCategories, Importer
                                           )

from .format import VelodynePointsPath


class VelodynePointsExtractor(SourceExtractor):
    _SUPPORTED_SHAPES = ('cuboid_3d')

    def __init__(self, path, subset=None):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        images_dir = ''
        if osp.isdir(osp.join(rootpath, VelodynePointsPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, VelodynePointsPath.IMAGES_DIR)
        self._images_dir = images_dir
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        super().__init__(subset=subset)

        items, categories = self._parse(path)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    @classmethod
    def _parse(cls, path):
        import xml.etree.ElementTree as ET
        tree = ET.parse(path)
        root = tree.getroot()
        shapes = {}
        shape = {"points": []}
        labels = OrderedDict()
        label = {}
        items = OrderedDict()
        categories = {}
        point_tags = ["h", "w", "l", "tx", "ty", "tz", "rx", "ry", "rz"]

        for elem in root.iter():
            if elem.tag == "objectType":
                shape["label"] = elem.text
                label["name"] = elem.text
            elif elem.tag in point_tags:
                shape['points'].append(float(elem.text))
            elif elem.tag == "first_frame":
                shape['frame'] = int(elem.text)
            elif elem.tag == "occlusion_kf":
                shape["occluded"] = 1 if int(elem.text) else 0
            elif elem.tag == "finished":
                for _ in range(7):
                    shape['points'].append(float(0.0))
                shape["type"] = "cuboid"
                label["attributes"] = {}
                labels[label['name']] = label["attributes"]

                shapes.update({len(shapes): shape})
                shape = {"points": []}

        common_attrs = ["occluded"]
        label_cat = LabelCategories(attributes=common_attrs)

        for label, attrs in labels.items():
            label_cat.add(label, attributes=attrs)

        categories[AnnotationType.label] = label_cat

        for shape in shapes.values():
            frame_desc = items.get(shape['frame'], {'annotations': []})
            frame_desc['annotations'].append(
                cls._parse_shape_ann(shape, categories))
            items[shape['frame']] = frame_desc

        return items, categories

    @classmethod
    def _parse_shape_ann(cls, ann, categories):
        ann_id = ann.get('id', 0)
        ann_type = ann['type']

        attributes = ann.get('attributes') or {}
        if 'occluded' in categories[AnnotationType.label].attributes:
            attributes['occluded'] = ann.get('occluded', 0)

        group = ann.get('group', 0)

        label = ann.get('label')
        label_id = categories[AnnotationType.label].find(label)[0]

        z_order = ann.get('z_order', 0)
        points = ann.get('points', [])

        if ann_type == "cuboid":
            return Cuboid3D(points, label=label_id, z_order=z_order,
                          id=ann_id, attributes=attributes, group=group)
        else:
            raise NotImplementedError("Unknown annotation type '%s'" % ann_type)

    def _load_items(self, parsed):
        for frame_id, item_desc in parsed.items():
            name = item_desc.get('name', 'frame_%06d.pcd' % int(frame_id))

            parsed[frame_id] = DatasetItem(id=osp.splitext(name)[0],
                                           subset=self._subset, related_images=[],
                                           annotations=item_desc.get('annotations'),
                                           attributes={'frame': int(frame_id)})
        return parsed


class VelodynePointsImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.xml', 'velodyne_points')
