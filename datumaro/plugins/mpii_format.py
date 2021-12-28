# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import scipy.io as spio

from datumaro.components.annotation import (
    Bbox, LabelCategories, Points, PointsCategories,
)
from datumaro.components.errors import DatasetImportError
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Importer, SourceExtractor,
)
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image

MPII_POINTS_LABELS = [
    'r_ankle',
    'r_knee',
    'r_hip',
    'l_hip',
    'l_knee',
    'l_ankle',
    'pelvis',
    'thorax',
    'upper_neck',
    'head top',
    'r_wrist',
    'r_elbow',
    'r_shoulder',
    'l_shoulder',
    'l_elbow',
    'l_wrist'
]

MPII_POINTS_JOINTS = [
    (0, 1), (1, 2), (2, 6), (3, 4),
    (3, 6), (4, 5), (6, 7), (7, 8),
    (8, 9), (8, 12), (8, 13), (10, 11),
    (11, 12), (13, 14), (14, 15)
]

class MpiiExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()

        self._categories = {
            AnnotationType.label: LabelCategories.from_iterable(['human']),
            AnnotationType.points: PointsCategories.from_iterable(
                [(0, MPII_POINTS_LABELS, MPII_POINTS_JOINTS)])
        }

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        root_dir = osp.dirname(path)

        data = spio.loadmat(path, struct_as_record=False,
            squeeze_me=True).get('RELEASE', {})
        data = getattr(data, 'annolist', [])

        for item in data:
            image = ''
            annotations = []
            group_num = 1

            image = getattr(item, 'image', '')
            if isinstance(image, spio.matlab.mio5_params.mat_struct):
                image = getattr(image, 'name', '')
            else:
                raise DatasetImportError("The item '%s' has not "
                    "information about the image" % item)

            values = getattr(item, 'annorect', [])
            if isinstance(values, spio.matlab.mio5_params.mat_struct):
                values = [values]

            for val in values:
                x1 = None
                x2 = None
                y1 = None
                y2 = None
                keypoints = {}
                is_visible = {}
                attributes = {}

                scale = getattr(val, 'scale', 0.0)
                if isinstance(scale, float):
                    attributes['scale'] = scale

                objpos = getattr(val, 'objpos', None)
                if isinstance(objpos, spio.matlab.mio5_params.mat_struct):
                    attributes['center'] = [getattr(objpos, 'x', 0), getattr(objpos, 'y', 0)]

                annopoints = getattr(val, 'annopoints', None)
                if isinstance(annopoints, spio.matlab.mio5_params.mat_struct) and \
                        not isinstance(getattr(annopoints, 'point'),
                            spio.matlab.mio5_params.mat_struct):
                    for point in getattr(annopoints, 'point'):
                        point_id = getattr(point, 'id')
                        keypoints[point_id] = [getattr(point, 'x', 0), getattr(point, 'y', 0)]
                        is_visible[point_id] = getattr(point, 'is_visible', 1)
                        if not isinstance(is_visible[point_id], int):
                            is_visible[point_id] = 1

                x1 = getattr(val, 'x1', None)
                if not isinstance(x1, (int, float)):
                    x1 = None

                x2 = getattr(val, 'x2', None)
                if not isinstance(x2, (int, float)):
                    x2 = None

                y1 = getattr(val, 'y1', None)
                if not isinstance(y1, (int, float)):
                    y1 = None

                y2 = getattr(val, 'y2', None)
                if not isinstance(y2, (int, float)):
                    y2 = None

                if keypoints:
                    points = [0] * (2 * len(keypoints))
                    vis = [0] * len(keypoints)

                    keypoints = sorted(keypoints.items(), key=lambda x: x[0])
                    for i, (key, point) in enumerate(keypoints):
                        points[2 * i] = point[0]
                        points[2 * i + 1] = point[1]
                        vis[i] = is_visible.get(key, 1)

                    annotations.append(Points(points, vis, label=0, group=group_num,
                        attributes=attributes))

                if x1 is not None and x2 is not None \
                        and y1 is not None and y2 is not None:

                    annotations.append(Bbox(x1, y1, x2 - x1, y2 - y1,
                        label=0, group=group_num))

                group_num += 1

            item_id = osp.splitext(image)[0]

            items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                image=Image(path=osp.join(root_dir, image)),
                annotations=annotations)

        return items

class MpiiImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.mat', 'mpii')

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file('*.mat')
