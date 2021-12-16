# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os.path as osp

import numpy as np

from datumaro.components.annotation import Points
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image


class MpiiPath:
    ANNO_FILE_PREFIX = 'annot/mpii_'

class MpiiExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(subset=osp.splitext(osp.basename(path))[0].
            split('_', maxsplit=1)[1])

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        with open(path) as f:
            for ann in json.load(f):
                item_id = osp.splitext(ann.get('img_paths'))[0]

                center = ann['objpos']
                scale = float(ann['scale_provided'])
                keypoints = np.array(ann['joint_self'])
                keypoints = keypoints.reshape(keypoints.shape[0] * keypoints.shape[1])
                points = [p for i, p in enumerate(keypoints) if i % 3 != 2]
                vis = keypoints[2::3]
                vis = [int(val) for val in vis]

                root_dir = osp.dirname(osp.dirname(path))

                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=Image(path=osp.join(root_dir, ann.get('img_paths'))),
                    annotations=[Points(points, vis, attributes={'center': center,
                        'scale': scale})])

        return items

class MpiiImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.json', 'mpii')

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(MpiiPath.ANNO_FILE_PREFIX + '*' + '.json')
