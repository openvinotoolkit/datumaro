
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os

from collections import OrderedDict
import os.path as osp

from datumaro.components.extractor import (SourceExtractor, DatasetItem,
    AnnotationType,Cuboid3D,
    LabelCategories, Importer
)
from datumaro.util.image import Image

from .format import PointCloudPath


class PointCloudExtractor(SourceExtractor):
    _SUPPORTED_SHAPES = "cuboid"

    def __init__(self, path, subset=None):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        images_dir = ''
        if osp.isdir(osp.join(rootpath, PointCloudPath.ANNNOTATION_DIR)):
            images_dir = osp.join(rootpath, PointCloudPath.ANNNOTATION_DIR)
        self._images_dir = images_dir

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        items, categories = self._parse(path)
        super().__init__(subset=subset, categories=categories)
        self.set_items(list(self._load_items(items).values()))

    @classmethod
    def _parse(cls, path):
        meta = {}
        path = osp.abspath(path)
        items = OrderedDict()
        categories = {}
        mapping = {}

        if osp.basename(path) == "key_id_map.json":
            with open(path, "r") as f:
                mapping = json.load(f)

        meta_path = osp.abspath(osp.join(osp.dirname(path), "meta.json"))

        if osp.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        data_dir = osp.join(osp.dirname(path), PointCloudPath.DEFAULT_DIR, PointCloudPath.ANNNOTATION_DIR)

        labels = {}
        for _, _, files in os.walk(data_dir):
            for file in files:
                with open(osp.join(data_dir, file), "r") as f:
                    figure_data = json.load(f)

                common_attrs = ["occluded"]
                label_cat = LabelCategories(attributes=common_attrs)
                if meta:
                    for label in meta["classes"]:
                        attrs = []
                        for obj in figure_data["objects"]:
                            if obj["classTitle"] == label['title'] and obj["tags"]:

                                for tag in obj["tags"]:
                                    attrs.append(tag["name"])

                        label_cat.add(label['title'], attributes=attrs)

                categories[AnnotationType.label] = label_cat

                for label in figure_data["objects"]:
                    labels.update({label["key"]: label["classTitle"]})

                group = 0
                z_order = 0

                for figure in figure_data["figures"]:
                    attributes = {}
                    anno_points = []
                    geometry_type = ["position", "rotation", "dimensions"]
                    for geo in geometry_type:
                        anno_points.extend(float(i) for i in figure["geometry"][geo].values())

                    for _ in range(7):
                        anno_points.append(0.0)

                    map_id = mapping["figures"][figure['key']]
                    label = labels[figure.get("objectKey")]

                    for obj in figure_data["objects"]:
                        if obj["key"] == figure.get("objectKey"):
                            for tag in obj["tags"]:
                                if obj["key"] == figure.get("objectKey"):
                                    if tag["value"] == "true":
                                        tag["value"] = True
                                    elif tag["value"] == "false":
                                        tag["value"] = False
                                    attributes.update({tag["name"]: tag["value"]})

                    label = categories[AnnotationType.label].find(label)[0]

                    shape = Cuboid3D(anno_points, label=label, z_order=z_order,
                                   id=map_id, attributes=attributes, group=group)

                    frame = mapping["videos"][figure_data['key']]
                    frame_desc = items.get(frame, {'annotations': []})

                    frame_desc['annotations'].append(shape)
                    items[frame] = frame_desc
        return items, categories

    def _load_items(self, parsed):
        for frame_id, item_desc in parsed.items():
            name = item_desc.get('name', 'frame_%06d.png' % int(frame_id))
            image = osp.join(self._images_dir, name)
            image_size = (item_desc.get('height'), item_desc.get('width'))
            if all(image_size):
                image = Image(path=image, size=tuple(map(int, image_size)))

            parsed[frame_id] = DatasetItem(id=osp.splitext(name)[0],
                                           subset=self._subset, image=image,
                                           annotations=item_desc.get('annotations'),
                                           attributes={'frame': int(frame_id)})

        return parsed


class PointCloudImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        sources = cls._find_sources_recursive(path, '.json', 'point_cloud')
        sources = [source for source in sources if osp.basename(source["url"]) != "meta.json"]

        return sources
