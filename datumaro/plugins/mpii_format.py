# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import scipy.io as spio

from datumaro.components.annotation import (
    Bbox, LabelCategories, Points, PointsCategories,
)
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

MPI_POINTS_JOINTS = [
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
                [(0, MPII_POINTS_LABELS, MPI_POINTS_JOINTS)])
        }

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        root_dir = osp.dirname(path)

        data = spio.loadmat(path, struct_as_record=False, squeeze_me=True).get('RELEASE', {})
        data = data.__dict__['annolist']

        for item in data:
            image = ''
            annotations = []
            group_num = 1
            for mat_val in item._fieldnames:
                values = item.__dict__[mat_val]

                if mat_val == 'image':
                    image = values.__dict__['name']

                elif mat_val == 'annorect':
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
                        for anno_mat in val._fieldnames:
                            anno = val.__dict__[anno_mat]
                            if anno_mat == 'scale' and isinstance(anno, float):
                                attributes['scale'] = anno

                            elif anno_mat == 'objpos' and \
                                    isinstance(anno, spio.matlab.mio5_params.mat_struct):
                                attributes['center'] = [anno.__dict__['x'], anno.__dict__['y']]

                            elif anno_mat == 'annopoints' and \
                                isinstance(anno, spio.matlab.mio5_params.mat_struct) and \
                                not isinstance(anno.__dict__['point'],
                                               spio.matlab.mio5_params.mat_struct):

                                for point in anno.__dict__['point']:
                                    point_id = point.__dict__['id']
                                    keypoints[point_id] = [point.__dict__['x'], point.__dict__['y']]
                                    is_visible[point_id] = point.__dict__['is_visible']
                                    if not isinstance(is_visible[point_id], int):
                                        is_visible[point_id] = 1

                            elif anno_mat == 'x1' and \
                                    (isinstance(anno, float) or isinstance(anno, int)):
                                x1 = anno

                            elif anno_mat == 'x2' and \
                                    (isinstance(anno, float) or isinstance(anno, int)):
                                x2 = anno

                            elif anno_mat == 'y1' and \
                                    (isinstance(anno, float) or isinstance(anno, int)):
                                y1 = anno

                            elif anno_mat == 'y2' and \
                                    (isinstance(anno, float) or isinstance(anno, int)):
                                y2 = anno

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
