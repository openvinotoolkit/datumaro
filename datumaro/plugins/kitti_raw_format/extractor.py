
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from defusedxml import ElementTree as ET
import os.path as osp

from datumaro.components.extractor import (SourceExtractor, DatasetItem,
    AnnotationType, Cuboid3d, LabelCategories, Importer)
from datumaro.util.image import find_images

from .format import KittiRawPath, OcclusionStates


class KittiRawExtractor(SourceExtractor):
    # http://www.cvlibs.net/datasets/kitti/raw_data.php
    # https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_raw_data.zip
    # Check cpp header implementation for field meaning

    def __init__(self, path, subset=None):
        assert osp.isfile(path), path
        rootpath = osp.dirname(path)
        image_dir = ''
        for p in os.listdir(rootpath):
            if p.lower().startswith(KittiRawPath.IMG_DIR_PREFIX) and \
                    p.lower() != KittiRawPath.IMG_DIR_PREFIX and \
                    osp.isdir(p):
                image_dir = osp.join(rootpath, p)
        self._image_dir = image_dir
        self._path = path

        super().__init__(subset=subset)

        items, categories = self._parse(path)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    @classmethod
    def _parse(cls, path):
        tracks = []
        track = None
        shape = None
        attr = None
        labels = {}
        point_tags = {'tx', 'ty', 'tz', 'rx', 'ry', 'rz'}

        tree = ET.iterparse(path, events=("start", "end"))
        for ev, elem in tree:
            if ev == "start":
                if elem.tag == 'item':
                    if track is None:
                        track = {
                            'shapes': [],
                            'scale': {},
                            'label': None,
                            'attributes': {},
                            'start_frame': None,
                            'length': None,
                        }
                    else:
                        shape = {
                            'points': {},
                            'attributes': {},
                            'occluded': None,
                            'occluded_kf': False,
                        }

                elif elem.tag == 'attribute':
                    attr = {}

            elif ev == "end":
                if elem.tag == 'item':
                    assert track is not None

                    if shape:
                        track['shapes'].append(shape)
                        shape = None
                    else:
                        assert track['length'] == len(track['shapes'])

                        if track['label']:
                            labels.setdefault(track['label'], set())

                            for a in track['attributes']:
                                labels[track['label']].add(a)

                            for s in track['shapes']:
                                for a in s['attributes']:
                                    labels[track['label']].add(a)

                        tracks.append(track)
                        track = None

                # track tags
                elif track and elem.tag == 'objectType':
                    track['label'] = elem.text
                elif track and elem.tag in {'h', 'w', 'l'}:
                    track['scale'][elem.tag] = float(elem.text)
                elif track and elem.tag == 'first_frame':
                    track['start_frame'] = int(elem.text)
                elif track and elem.tag == 'count' and track:
                    track['length'] = int(elem.text)

                # pose tags
                elif shape and elem.tag in point_tags:
                    shape['points'][elem.tag] = float(elem.text)
                elif shape and elem.tag == 'occlusion':
                    shape['occluded'] = OcclusionStates(int(elem.text))
                elif shape and elem.tag == 'occlusion_kf':
                    shape['occluded_kf'] = elem.text == '1'

                # common tags
                elif attr is not None and elem.tag == 'name':
                    attr['name'] = elem.text
                elif attr is not None and elem.tag == 'value':
                    attr['value'] = elem.text
                elif attr is not None and elem.tag == 'attribute':
                    if shape:
                        shape['attributes'][attr['name']] = elem.tag
                    else:
                        track['attributes'][attr['name']] = elem.tag
                    attr = None

        if track is not None or shape is not None or attr is not None:
            raise Exception("Failed to parse anotations from '%s'" % path)

        common_attrs = ['occluded']
        label_cat = LabelCategories(attributes=common_attrs)
        for label, attrs in sorted(labels.items(), key=lambda e: e[0]):
            label_cat.add(label, attributes=attrs)

        categories = {AnnotationType.label: label_cat}

        items = {}
        for idx, track in enumerate(tracks):
            track_id = idx + 1
            for i, ann in enumerate(
                    cls._parse_track(track_id, track, categories)):
                frame_desc = items.setdefault(track['start_frame'] + i,
                    {'annotations': []})
                frame_desc['annotations'].append(ann)

        return items, categories

    @classmethod
    def _parse_attr(cls, value):
        if value == 'true':
            return True
        elif value == 'false':
            return False
        else:
            return value

    @classmethod
    def _parse_track(cls, track_id, track, categories):
        common_attrs = { k: cls._parse_attr(v)
            for k, v in track['attributes'].items() }
        scale = [track['scale'][k] for k in ['w', 'h', 'l']]
        label = categories[AnnotationType.label].find(track['label'])[0]

        kf_occluded = False
        for shape in track['shapes']:
            occluded = shape['occluded'] in {
                OcclusionStates.FULLY, OcclusionStates.PARTLY}
            if shape['occluded_kf']:
                kf_occluded = occluded
            elif shape['occluded'] == OcclusionStates.OCCLUSION_UNSET:
                occluded = kf_occluded

            local_attrs = { k: cls._parse_attr(v)
                for k, v in shape['attributes'].items() }
            local_attrs['occluded'] = occluded
            local_attrs['track_id'] = track_id
            attrs = dict(common_attrs)
            attrs.update(local_attrs)

            position = [shape['points'][k] for k in ['tx', 'ty', 'tz']]
            rotation = [shape['points'][k] for k in ['rx', 'ry', 'rz']]

            yield Cuboid3d(position, rotation, scale, label=label,
                attributes=attrs)

    def _load_items(self, parsed):
        image_dir = osp.join(self._image_dir, KittiRawPath.IMG_DIR_PREFIX)
        if osp.isdir(image_dir):
            images = { osp.splitext(osp.relpath(p, image_dir))[0]: p
                for p in find_images(image_dir, recursive=True) }
        else:
            images = {}

        items = {}
        for frame_id, item_desc in parsed.items():
            name = item_desc.get('name', '%010d' % int(frame_id))
            related_images = []
            if name in images:
                related_images.append(images[name])

            items[frame_id] = DatasetItem(id=name, subset=self._subset,
                pcd=osp.join(osp.dirname(self._path),
                    KittiRawPath.PCD_DIR, name + '.pcd'),
                related_images=related_images,
                annotations=item_desc.get('annotations'),
                attributes={'frame': int(frame_id)})
        return items


class KittiRawImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.xml', 'kitti_raw')
