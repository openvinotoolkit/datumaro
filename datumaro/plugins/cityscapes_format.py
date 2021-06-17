
# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict
from enum import Enum, auto
from glob import iglob

import numpy as np

from datumaro.components.converter import Converter
from datumaro.components.extractor import (AnnotationType, CompiledMask,
    DatasetItem, Importer, LabelCategories, Mask,
    MaskCategories, SourceExtractor)
from datumaro.util import str_to_bool
from datumaro.util.annotation_util import make_label_id_mapping
from datumaro.util.image import save_image, load_image
from datumaro.util.mask_tools import generate_colormap, paint_mask


CityscapesLabelMap = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('egovehicle', (0, 0, 0)),
    ('rectificationborder', (0, 0, 0)),
    ('outofroi', (0, 0, 0)),
    ('static', (0, 0, 0)),
    ('dynamic', (111, 74, 0)),
    ('ground', (81, 0, 81)),
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('parking', (250, 170, 160)),
    ('railtrack', (230, 150, 140)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('guardrail', (180, 165, 180)),
    ('bridge', (150, 100, 100)),
    ('tunnel', (150, 120, 90)),
    ('pole', (153, 153, 153)),
    ('polegroup', (153, 153, 153)),
    ('trafficlight', (250, 170, 30)),
    ('trafficsign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255, 0, 0)),
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)),
    ('caravan', (0, 0, 90)),
    ('trailer', (0, 0, 110)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32)),
    ('licenseplate', (0, 0, 142)),
])

class CityscapesPath:
    GT_FINE_DIR = 'gtFine'
    IMGS_FINE_DIR = 'imgsFine'
    ORIGINAL_IMAGE_DIR = 'leftImg8bit'
    ORIGINAL_IMAGE = '_%s.png' % ORIGINAL_IMAGE_DIR
    INSTANCES_IMAGE = '_instanceIds.png'
    COLOR_IMAGE = '_color.png'
    LABELIDS_IMAGE = '_labelIds.png'

    LABELMAP_FILE = 'label_colors.txt'

def make_cityscapes_categories(label_map=None):
    if label_map is None:
        label_map = CityscapesLabelMap

    categories = {}
    label_categories = LabelCategories()
    for label in label_map:
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

class CityscapesExtractor(SourceExtractor):
    def __init__(self, path, subset=None):
        assert osp.isdir(path), path
        self._path = path

        if not subset:
            subset = osp.splitext(osp.basename(path))[0]
        self._subset = subset
        super().__init__(subset=subset)

        self._categories = self._load_categories(osp.join(self._path, '../../../'))
        self._items = list(self._load_items().values())

    def _load_categories(self, path):
        label_map = None
        label_map_path = osp.join(path, CityscapesPath.LABELMAP_FILE)
        if osp.isfile(label_map_path):
            label_map = parse_label_map(label_map_path)
        else:
            label_map = CityscapesLabelMap
        self._labels = [label for label in label_map]
        return make_cityscapes_categories(label_map)

    def _load_items(self):
        items = {}
        annotations_path = osp.normpath(osp.join(self._path, '../../../',
            CityscapesPath.GT_FINE_DIR, self._subset))

        for image_path in iglob(
                osp.join(self._path, '**', '*' + CityscapesPath.ORIGINAL_IMAGE),
                recursive=True):
            sample_id = osp.relpath(image_path, self._path) \
                .replace(CityscapesPath.ORIGINAL_IMAGE, '')
            anns = []
            instances_path = osp.join(annotations_path, sample_id + '_' +
                CityscapesPath.GT_FINE_DIR + CityscapesPath.INSTANCES_IMAGE)
            if osp.isfile(instances_path):
                instances_mask = load_image(instances_path, dtype=np.int32)
                segm_ids = np.unique(instances_mask)
                for segm_id in segm_ids:
                    if segm_id < 1000:
                        semanticId = segm_id
                        isCrowd = True
                        ann_id = segm_id
                    else:
                        semanticId = segm_id // 1000
                        isCrowd = False
                        ann_id = segm_id % 1000
                    anns.append(Mask(
                        image=self._lazy_extract_mask(instances_mask, segm_id),
                        label=semanticId, id=ann_id,
                        attributes = { 'is_crowd': isCrowd }))
            items[sample_id] = DatasetItem(id=sample_id, subset=self._subset,
                image=image_path, annotations=anns)
        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c


class CityscapesImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'cityscapes',
            dirname=osp.join(CityscapesPath.IMGS_FINE_DIR,
                CityscapesPath.ORIGINAL_IMAGE_DIR),
            max_depth=1)


class LabelmapType(Enum):
    cityscapes = auto()
    source = auto()

class CityscapesConverter(Converter):
    DEFAULT_IMAGE_EXT = '.png'

    @staticmethod
    def _get_labelmap(s):
        if osp.isfile(s):
            return s
        try:
            return LabelmapType[s].name
        except KeyError:
            import argparse
            raise argparse.ArgumentTypeError()

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)

        parser.add_argument('--apply-colormap', type=str_to_bool, default=True,
            help="Use colormap for class masks (default: %(default)s)")
        parser.add_argument('--label-map', type=cls._get_labelmap, default=None,
            help="Labelmap file path or one of %s" % \
                ', '.join(t.name for t in LabelmapType))
        return parser

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
            for item in subset:
                image_path = osp.join(CityscapesPath.IMGS_FINE_DIR,
                    CityscapesPath.ORIGINAL_IMAGE_DIR, subset_name,
                    item.id + CityscapesPath.ORIGINAL_IMAGE)
                if self._save_images:
                    self._save_image(item, osp.join(self._save_dir, image_path))

                common_folder_path = osp.join(CityscapesPath.GT_FINE_DIR,
                    subset_name)

                masks = [a for a in item.annotations
                    if a.type == AnnotationType.mask]
                if not masks:
                    continue

                common_image_name = item.id + '_' + CityscapesPath.GT_FINE_DIR

                compiled_class_mask = CompiledMask.from_instance_masks(masks,
                    instance_labels=[self._label_id_mapping(m.label)
                        for m in masks])
                color_mask_path = osp.join(common_folder_path,
                    common_image_name + CityscapesPath.COLOR_IMAGE)
                self.save_mask(osp.join(self._save_dir, color_mask_path),
                    compiled_class_mask.class_mask)

                labelids_mask_path = osp.join(common_folder_path,
                    common_image_name + CityscapesPath.LABELIDS_IMAGE)
                self.save_mask(osp.join(self._save_dir, labelids_mask_path),
                    compiled_class_mask.class_mask, apply_colormap=False,
                    dtype=np.int32)

                compiled_instance_mask = CompiledMask.from_instance_masks(masks,
                    instance_labels=[m.id if m.attributes.get('is_crowd', True)
                    else m.label * 1000 + m.id for m in masks])
                inst_path = osp.join(common_folder_path,
                    common_image_name + CityscapesPath.INSTANCES_IMAGE)
                self.save_mask(osp.join(self._save_dir, inst_path),
                    compiled_instance_mask.class_mask, apply_colormap=False,
                    dtype=np.int32)
        self.save_label_map()

    def save_label_map(self):
        path = osp.join(self._save_dir, CityscapesPath.LABELMAP_FILE)
        write_label_map(path, self._label_map)

    def _load_categories(self, label_map_source):
        if label_map_source == LabelmapType.cityscapes.name:
            # use the default Cityscapes colormap
            label_map = CityscapesLabelMap

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

        self._categories = make_cityscapes_categories(label_map)
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

    def save_mask(self, path, mask, colormap=None, apply_colormap=True,
        dtype=np.uint8):
        if self._apply_colormap and apply_colormap:
            if colormap is None:
                colormap = self._categories[AnnotationType.mask].colormap
            mask = paint_mask(mask, colormap)
        save_image(path, mask, create_dir=True, dtype=dtype)
