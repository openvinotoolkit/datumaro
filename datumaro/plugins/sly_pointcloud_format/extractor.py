# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import iglob
import json
import os.path as osp

from datumaro.components.extractor import (
    AnnotationType, Cuboid3d, DatasetItem, Importer, LabelCategories,
    SourceExtractor,
)
from datumaro.util.image import find_images

from .format import PointCloudPath


class SuperviselyPointCloudExtractor(SourceExtractor):
    NAME = 'sly_pointcloud'
    _SUPPORTED_SHAPES = 'cuboid'

    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Expected a path to 'meta.json', "
                "got '%s'" % path)

        rootdir = osp.abspath(osp.dirname(path))
        self._rootdir = rootdir

        super().__init__(subset=subset)

        items, categories = self._parse(rootdir)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    @classmethod
    def _parse(cls, rootpath):
        with open(osp.join(rootpath, PointCloudPath.KEY_ID_FILE),
                encoding='utf-8') as f:
            mapping = json.load(f)

        with open(osp.join(rootpath, PointCloudPath.META_FILE),
                encoding='utf-8') as f:
            meta = json.load(f)

        label_cat = LabelCategories()
        for label in meta.get('classes', []):
            label_cat.add(label['title'])

        tags = {}
        for tag in meta.get('tags', []):
            # See reference at:
            # https://github.com/supervisely/supervisely/blob/047e52ebe407cfee61464c1bd0beb9c906892253/supervisely_lib/annotation/tag_meta.py#L139
            tags[tag['name']] = tag

            applicable_to = tag.get('applicable_type', 'all')
            if applicable_to == 'imagesOnly':
                continue # an image attribute
            elif applicable_to not in {'all', 'objectsOnly'}:
                raise Exception("Unexpected tag 'applicable_type' value '%s'" % \
                    applicable_to)

            applicable_classes = tag.get('classes', [])
            if not applicable_classes:
                label_cat.attributes.add(tag['name'])
            else:
                for label_name in applicable_classes:
                    _, label = label_cat.find(label_name)
                    if label is None:
                        raise Exception("Unknown class for tag '%s'" % \
                            label_name)

                    label.attributes.add(tag['name'])

        categories = {AnnotationType.label: label_cat}

        def _get_label_attrs(label_id):
            attrs = set(label_cat.attributes)
            attrs.update(label_cat[label_id].attributes)
            return attrs

        def _parse_tag(tag):
            if tag['value'] == 'true':
                value = True
            elif tag['value'] == 'false':
                value = False
            else:
                value = tag['value']
            return value

        ann_dir = osp.join(rootpath,
            PointCloudPath.BASE_DIR, PointCloudPath.ANNNOTATION_DIR)
        items = {}
        for ann_file in iglob(osp.join(ann_dir, '**', '*.json'), recursive=True):
            with open(ann_file, encoding='utf-8') as f:
                ann_data = json.load(f)

            objects = {}
            for obj in ann_data['objects']:
                obj['id'] = mapping['objects'][obj['key']]
                objects[obj['key']] = obj

            frame_attributes = {'description': ann_data.get('description', '')}
            for tag in ann_data['tags']:
                frame_attributes[tag['name']] = _parse_tag(tag)

            frame = mapping['videos'][ann_data['key']]
            frame_desc = items.setdefault(frame, {
                'name': osp.splitext(osp.relpath(ann_file, ann_dir))[0],
                'annotations': [],
                'attributes': frame_attributes,
            })

            for figure in ann_data['figures']:
                geometry = {
                    dst_field: [float(figure['geometry'][src_field][axis])
                        for axis in ['x', 'y', 'z']
                    ]
                    for src_field, dst_field in {
                        'position': 'position',
                        'rotation': 'rotation',
                        'dimensions': 'scale'
                    }.items()
                }

                ann_id = mapping['figures'][figure['key']]

                obj = objects[figure['objectKey']]
                label = categories[AnnotationType.label].find(
                    obj['classTitle'])[0]

                attributes = {}
                attributes['track_id'] = obj['id']
                for tag in obj.get('tags', []):
                    attributes[tag['name']] = _parse_tag(tag)
                for attr in _get_label_attrs(label):
                    if attr in attributes:
                        continue
                    if tags[attr]['value_type'] == 'any_string':
                        value = ''
                    elif tags[attr]['value_type'] == 'oneof_string':
                        value = (tags[attr]['values'] or [''])[0]
                    elif tags[attr]['value_type'] == 'any_number':
                        value = 0
                    else:
                        value = None
                    attributes[attr] = value

                shape = Cuboid3d(**geometry, label=label,
                    id=ann_id, attributes=attributes)

                frame_desc['annotations'].append(shape)

        return items, categories

    def _load_items(self, parsed):
        for frame_id, frame_desc in parsed.items():
            pcd_name = frame_desc['name']
            name = osp.splitext(pcd_name)[0]
            pcd_path = osp.join(self._rootdir, PointCloudPath.BASE_DIR,
                PointCloudPath.POINT_CLOUD_DIR, pcd_name)
            assert pcd_path.endswith('.pcd'), pcd_path

            related_images_dir = osp.join(self._rootdir,
                PointCloudPath.BASE_DIR,
                PointCloudPath.RELATED_IMAGES_DIR, name + '_pcd')
            related_images = None
            if osp.isdir(related_images_dir):
                related_images = find_images(related_images_dir)

            parsed[frame_id] = DatasetItem(id=name, subset=self._subset,
                point_cloud=pcd_path, related_images=related_images,
                annotations=frame_desc.get('annotations'),
                attributes={'frame': int(frame_id), **frame_desc['attributes']})

        return parsed


class SuperviselyPointCloudImporter(Importer):
    NAME = 'sly_pointcloud'

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.json', 'sly_pointcloud',
            filename='meta')
