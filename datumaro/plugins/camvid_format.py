
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict
from enum import Enum

import numpy as np

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, CompiledMask,
    DatasetItem, Importer, LabelCategories, Mask,
    MaskCategories, SourceExtractor)
from datumaro.util import find, str_to_bool
from datumaro.util.annotation_util import make_label_id_mapping
from datumaro.util.image import save_image
from datumaro.util.mask_tools import lazy_mask, paint_mask, generate_colormap


CamvidLabelMap = OrderedDict([
    ('Void', (0, 0, 0)),
    ('Animal', (64, 128, 64)),
    ('Archway', (192, 0, 128)),
    ('Bicyclist', (0, 128, 192)),
    ('Bridge', (0, 128, 64)),
    ('Building', (128, 0, 0)),
    ('Car', (64, 0, 128)),
    ('CartLuggagePram', (64, 0, 192)),
    ('Child', (192, 128, 64)),
    ('Column_Pole', (192, 192, 128)),
    ('Fence', (64, 64, 128)),
    ('LaneMkgsDriv', (128, 0, 192)),
    ('LaneMkgsNonDriv', (192, 0, 64)),
    ('Misc_Text', (128, 128, 64)),
    ('MotorcycycleScooter', (192, 0, 192)),
    ('OtherMoving', (128, 64, 64)),
    ('ParkingBlock', (64, 192, 128)),
    ('Pedestrian', (64, 64, 0)),
    ('Road', (128, 64, 128)),
    ('RoadShoulder', (128, 128, 192)),
    ('Sidewalk', (0, 0, 192)),
    ('SignSymbol', (192, 128, 128)),
    ('Sky', (128, 128, 128)),
    ('SUVPickupTruck', (64, 128, 192)),
    ('TrafficCone', (0, 0, 64)),
    ('TrafficLight', (0, 64, 64)),
    ('Train', (192, 64, 128)),
    ('Tree', (128, 128, 0)),
    ('Truck_Bus', (192, 128, 192)),
    ('Tunnel', (64, 0, 64)),
    ('VegetationMisc', (192, 192, 0)),
    ('Wall', (64, 192, 0))
])

class CamvidPath:
    LABELMAP_FILE = 'label_colors.txt'
    SEGM_DIR = "annot"
    IMAGE_EXT = '.jpg'
    MASK_EXT = '.png'


def parse_label_map(path):
    if not path:
        return None

    label_map = OrderedDict()
    with open(path, 'r') as f:
        for line in f:
            # skip empty and commented lines
            line = line.strip()
            if not line or line and line[0] == '#':
                continue

            # color, name
            label_desc = line.strip().split()

            if 2 < len(label_desc):
                name = label_desc[3]
                color = tuple([int(c) for c in label_desc[:-1]])
            else:
                name = label_desc[0]
                color = None

            if name in label_map:
                raise ValueError("Label '%s' is already defined" % name)

            label_map[name] = color
    return label_map

def write_label_map(path, label_map):
    with open(path, 'w') as f:
        for label_name, label_desc in label_map.items():
            if label_desc:
                color_rgb = ' '.join(str(c) for c in label_desc)
            else:
                color_rgb = ''
            f.write('%s %s\n' % (color_rgb, label_name))

def make_camvid_categories(label_map=None):
    if label_map is None:
        label_map = CamvidLabelMap

    # There must always be a label with color (0, 0, 0) at index 0
    bg_label = find(label_map.items(), lambda x: x[1] == (0, 0, 0))
    if bg_label is not None:
        bg_label = bg_label[0]
    else:
        bg_label = 'background'
        if bg_label not in label_map:
            has_colors = any(v is not None for v in label_map.values())
            color = (0, 0, 0) if has_colors else None
            label_map[bg_label] = color
    label_map.move_to_end(bg_label, last=False)

    categories = {}
    label_categories = LabelCategories()
    for label, desc in label_map.items():
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v is not None for v in label_map.values())
    if not has_colors: # generate new colors
        colormap = generate_colormap(len(label_map))
    else: # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = { label_id(name): (desc[0], desc[1], desc[2])
            for name, desc in label_map.items() }
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories
    return categories


class CamvidExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isfile(path), path
        self._path = path
        self._dataset_dir = osp.dirname(path)
        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        self._categories = self._load_categories(self._dataset_dir)
        self._items = list(self._load_items(path).values())

    def _load_categories(self, path):
        label_map = None
        label_map_path = osp.join(path, CamvidPath.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = CamvidLabelMap
        self._labels = [label for label in label_map]
        return make_camvid_categories(label_map)

    def _load_items(self, path):
        items = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                objects = line.split()
                image = objects[0]
                item_id = ('/'.join(image.split('/')[2:]))[:-len(CamvidPath.IMAGE_EXT)]
                image_path = osp.join(self._dataset_dir,
                    (image, image[1:])[image[0] == '/'])
                item_annotations = []
                if 1 < len(objects):
                    gt = objects[1]
                    gt_path = osp.join(self._dataset_dir,
                        (gt, gt[1:]) [gt[0] == '/'])
                    inverse_cls_colormap = \
                        self._categories[AnnotationType.mask].inverse_colormap
                    mask = lazy_mask(gt_path, inverse_cls_colormap)
                    # loading mask through cache
                    mask = mask()
                    classes = np.unique(mask)
                    labels = self._categories[AnnotationType.label]._indices
                    labels = { labels[label_name]: label_name
                        for label_name in labels }
                    for label_id in classes:
                        if labels[label_id] in self._labels:
                            image = self._lazy_extract_mask(mask, label_id)
                            item_annotations.append(Mask(image=image, label=label_id))
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=image_path, annotations=item_annotations)
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class CamvidImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.txt', 'camvid',
            file_filter=lambda p: osp.basename(p) != CamvidPath.LABELMAP_FILE)


LabelmapType = Enum('LabelmapType', ['camvid', 'source'])

class CamvidConverter(Converter):
    DEFAULT_IMAGE_EXT = CamvidPath.IMAGE_EXT

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument('--apply-colormap', type=str_to_bool, default=True,
            help="Use colormap for class masks (default: %(default)s)")
        parser.add_argument('--label-map', type=cls._get_labelmap, default=None,
            help="Labelmap file path or one of %s" % \
                ', '.join(t.name for t in LabelmapType))

    def __init__(self, extractor, save_dir,
            apply_colormap=True, label_map=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        self._apply_colormap = apply_colormap

        if label_map is None:
            label_map = LabelmapType.source.name
        self._load_categories(label_map)

    def apply(self):
        os.makedirs(self._save_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            segm_list = {}
            for item in subset:
                image_path = self._make_image_filename(item, subdir=subset_name)
                if self._save_images:
                    self._save_image(item, osp.join(self._save_dir, image_path))

                masks = [a for a in item.annotations
                    if a.type == AnnotationType.mask]

                if masks:
                    compiled_mask = CompiledMask.from_instance_masks(masks,
                        instance_labels=[self._label_id_mapping(m.label)
                            for m in masks])

                    mask_path = osp.join(subset_name + CamvidPath.SEGM_DIR,
                        item.id + CamvidPath.MASK_EXT)
                    self.save_segm(osp.join(self._save_dir, mask_path),
                        compiled_mask.class_mask)
                    segm_list[item.id] = (image_path, mask_path)
                else:
                    segm_list[item.id] = (image_path, '')

            self.save_segm_lists(subset_name, segm_list)
        self.save_label_map()

    def save_segm(self, path, mask, colormap=None):
        if self._apply_colormap:
            if colormap is None:
                colormap = self._categories[AnnotationType.mask].colormap
            mask = paint_mask(mask, colormap)
        save_image(path, mask, create_dir=True)

    def save_segm_lists(self, subset_name, segm_list):
        if not segm_list:
            return

        ann_file = osp.join(self._save_dir, subset_name + '.txt')
        with open(ann_file, 'w') as f:
            for (image_path, mask_path) in segm_list.values():
                f.write('/%s %s\n' % (
                    image_path.replace('\\', '/'),
                    mask_path.replace('\\', '/'))
                )

    def save_label_map(self):
        path = osp.join(self._save_dir, CamvidPath.LABELMAP_FILE)
        labels = self._extractor.categories()[AnnotationType.label]._indices
        if len(self._label_map) > len(labels):
            self._label_map.pop('background')
        write_label_map(path, self._label_map)

    def _load_categories(self, label_map_source):
        if label_map_source == LabelmapType.camvid.name:
            # use the default Camvid colormap
            label_map = CamvidLabelMap

        elif label_map_source == LabelmapType.source.name and \
                AnnotationType.mask not in self._extractor.categories():
            # generate colormap for input labels
            labels = self._extractor.categories() \
                .get(AnnotationType.label, LabelCategories())
            label_map = OrderedDict((item.name, None)
                for item in labels.items)

        elif label_map_source == LabelmapType.source.name and \
                AnnotationType.mask in self._extractor.categories():
            # use source colormap
            labels = self._extractor.categories()[AnnotationType.label]
            colors = self._extractor.categories()[AnnotationType.mask]
            label_map = OrderedDict()
            for idx, item in enumerate(labels.items):
                color = colors.colormap.get(idx)
                if color is not None:
                    label_map[item.name] = color

        elif isinstance(label_map_source, dict):
            label_map = OrderedDict(
                sorted(label_map_source.items(), key=lambda e: e[0]))

        elif isinstance(label_map_source, str) and osp.isfile(label_map_source):
            label_map = parse_label_map(label_map_source)

        else:
            raise Exception("Wrong labelmap specified, "
                "expected one of %s or a file path" % \
                ', '.join(t.name for t in LabelmapType))

        self._categories = make_camvid_categories(label_map)
        self._label_map = label_map
        self._label_id_mapping = self._make_label_id_map()

    def _make_label_id_map(self):
        map_id, id_mapping, src_labels, dst_labels = make_label_id_mapping(
            self._extractor.categories().get(AnnotationType.label),
            self._categories[AnnotationType.label])

        void_labels = [src_label for src_id, src_label in src_labels.items()
            if src_label not in dst_labels]
        if void_labels:
            log.warning("The following labels are remapped to background: %s" %
                ', '.join(void_labels))
        log.debug("Saving segmentations with the following label mapping: \n%s" %
            '\n'.join(["#%s '%s' -> #%s '%s'" %
                (
                    src_id, src_label, id_mapping[src_id],
                    self._categories[AnnotationType.label] \
                        .items[id_mapping[src_id]].name
                )
                for src_id, src_label in src_labels.items()
            ])
        )

        return map_id
