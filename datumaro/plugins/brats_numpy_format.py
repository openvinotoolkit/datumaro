# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import pickle  # nosec - disable B403:import_pickle check - fixed

import numpy as np
import numpy.core.multiarray

from datumaro.components.annotation import (
    AnnotationType, Cuboid3d, LabelCategories, Mask,
)
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "numpy.core.multiarray" and \
                name in PickleLoader.safe_numpy:
            return getattr(numpy.core.multiarray, name)
        elif module == 'numpy' and name in PickleLoader.safe_numpy:
            return getattr(numpy, name)
        raise pickle.UnpicklingError("Global '%s.%s' is forbidden"
            % (module, name))

class PickleLoader():
    safe_numpy = {
        'dtype',
        'ndarray',
        '_reconstruct',
    }

    def restricted_load(s):
        return RestrictedUnpickler(s, encoding='latin1').load()

class BratsNumpyPath:
    IDS_FILE = 'val_ids.p'
    BOXES_FILE = 'val_brain_bbox.p'
    LABELS_FILE = 'labels'
    DATA_SUFFIX = '_data_cropped'
    LABEL_SUFFIX = '_label_cropped'


class BratsNumpyExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()

        self._root_dir = osp.dirname(path)
        self._categories = self._load_categories()
        self._items = list(self._load_items(path).values())

    def _load_categories(self):
        label_cat = LabelCategories()

        labels_path = osp.join(self._root_dir, BratsNumpyPath.LABELS_FILE)
        if osp.isfile(labels_path):
            with open(labels_path, encoding='utf-8') as f:
                for line in f:
                    label_cat.add(line.strip())

        return { AnnotationType.label: label_cat }

    def _load_items(self, path):
        items = {}

        with open(path, 'rb') as f:
            ids = PickleLoader.restricted_load(f)

        boxes = None
        boxes_file = osp.join(self._root_dir, BratsNumpyPath.BOXES_FILE)
        if osp.isfile(boxes_file):
            with open(boxes_file, 'rb') as f:
                boxes = PickleLoader.restricted_load(f)

        for i, item_id in enumerate(ids):
            data_file = osp.join(self._root_dir, item_id + BratsNumpyPath.DATA_SUFFIX + '.npy')
            image = None
            if osp.isfile(data_file):
                image = np.load(data_file)[0].transpose()

            anno = []
            label_file = osp.join(self._root_dir, item_id + BratsNumpyPath.LABEL_SUFFIX + '.npy')
            if osp.isfile(label_file):
                mask = np.load(label_file)[0].transpose()
                classes = np.unique(mask)

                for class_id in classes:
                    anno.append(Mask(image=self._lazy_extract_mask(mask, class_id),
                        label=class_id))

            if boxes is not None:
                box = boxes[i]
                anno.append(Cuboid3d(position=list(box[0]), rotation=list(box[1])))

            items[item_id] = DatasetItem(id=item_id, image=image, annotations=anno)

        return items

    @staticmethod
    def _lazy_extract_mask(mask, c):
        return lambda: mask == c

class BratsNumpyImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(BratsNumpyPath.IDS_FILE)

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '', 'brats_numpy',
            filename=BratsNumpyPath.IDS_FILE)
