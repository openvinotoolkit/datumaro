
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import glob
import os
import os.path as osp
import numpy as np
from collections import OrderedDict
from enum import Enum

from datumaro.components.extractor import (SourceExtractor, DatasetItem,
    AnnotationType, Mask, CompiledMask, Importer, LabelCategories, MaskCategories
)
from datumaro.components.converter import Converter
from datumaro.util.mask_tools import paint_mask, lazy_mask
from datumaro.util.image import save_image

CamvidLabelMap = OrderedDict([
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
    ('Void', (0, 0, 0)),
    ('Wall', (64, 192, 0))
])

class CamvidPath:
    LABELMAP_FILE = 'label_colors.txt'
    SEGM_DIR = "annot"
    IMAGE_EXT = '.png'


def parse_label_map(path):
    if not path:
        return None
    label_map = OrderedDict()
    with open(path, 'r') as f:
        for line in f:
            if not line:
                continue
            label_desc = line.strip().split()
            color  = (int(label_desc[0]), int(label_desc[1]),
                int(label_desc[2]))
            name = label_desc[3]
            label_map[name] = color
    return label_map

def write_label_map(path, label_map):
    with open(path, 'w') as f:
        for label_name, label_desc in label_map.items():
            f.write('%s %s %s %s\n' % (label_desc[0],
                label_desc[1], label_desc[2], label_name))

def make_camvid_categories(label_map=None):
    if label_map is None:
        label_map = CamvidLabelMap
    categories = {}
    label_categories = LabelCategories()
    for label, desc in label_map.items():
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories
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

    @staticmethod
    def _load_categories(path):
        label_map = None
        label_map_path = osp.join(path, CamvidPath.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        return make_camvid_categories(label_map)

    def _load_items(self, path):
        items = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                image, gt = line.split()
                item_id = ('/'.join(gt.split('/')[2:]))[:-len(CamvidPath.IMAGE_EXT)]
                image_path = osp.join(self._dataset_dir, image[1:])
                gt_path = osp.join(self._dataset_dir, gt[1:])
                inverse_cls_colormap = \
                    self._categories[AnnotationType.mask].inverse_colormap
                mask = lazy_mask(gt_path, inverse_cls_colormap)
                mask = mask()
                classes = np.unique(mask)
                item_annotations = []
                for label_id in classes:
                    image = self._lazy_extract_mask(mask, label_id)
                    item_annotations.append(Mask(image=image, label=label_id))
                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=osp.join(image_path), annotations=item_annotations)
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class CamvidImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        subset_paths = [p for p in glob(osp.join(path, '*.txt'))
            if osp.basename(p) != CamvidPath.LABELMAP_FILE]
        sources = []
        for subset_path in subset_paths:
            sources += cls._find_sources_recursive(
                subset_path, '.txt', 'camvid')
        return sources


LabelmapType = Enum('LabelmapType', ['camvid', 'source'])

class CamvidConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    def __init__(self, extractor, save_dir,
            apply_colormap=True, label_map=None, **kwargs):
        super().__init__(extractor, save_dir, **kwargs)

        self._apply_colormap = apply_colormap

        if label_map is None:
            label_map = LabelmapType.source.name
        self._load_categories(label_map)

    def apply(self):
        subset_dir = self._save_dir
        os.makedirs(subset_dir, exist_ok=True)

        for subset_name, subset in self._extractor.subsets().items():
            segm_list = []
            for item in subset:
                masks = []
                for a in item.annotations:
                    if a.type == AnnotationType.mask:
                        masks.append(a)

                if masks:
                    compiled_mask = CompiledMask.from_instance_masks(masks)

                    self.save_segm(
                        osp.join(subset_dir, subset_name + CamvidPath.SEGM_DIR, item.id + CamvidPath.IMAGE_EXT),
                        compiled_mask.class_mask)

                    self._save_image(item,
                        osp.join(subset_dir, subset_name, item.id + CamvidPath.IMAGE_EXT))

                    segm_list.append(item.id)

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
            for item in segm_list:
                f.write('/%s/%s /%s/%s\n' % (subset_name, item + CamvidPath.IMAGE_EXT,
                    subset_name + CamvidPath.SEGM_DIR, item + CamvidPath.IMAGE_EXT))

    def save_label_map(self):
        path = osp.join(self._save_dir, CamvidPath.LABELMAP_FILE)
        write_label_map(path, self._label_map)

    def _load_categories(self, label_map_source):
        if label_map_source == LabelmapType.camvid.name:
            # use the default Camvid colormap
            label_map = CamvidLabelMap

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
