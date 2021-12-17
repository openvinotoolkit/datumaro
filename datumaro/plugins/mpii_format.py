# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox, Points, PointsCategories
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Importer, SourceExtractor,
)
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image


class MpiiPath:
    ANNOTATION_FILE = 'mpii_annotations.json'
    HEADBOXES_FILE = 'mpii_headboxes.npy'
    VISIBILITY_FILE = 'jnt_visible.npy'
    POS_GT_FILE = 'mpii_pos_gt.npy'

MpiiPointsCategories = [(0, ['r_ankle']), (1, ['r_knee']), (2, ['r_hip']),
    (3, ['l_hip']), (4, ['l_knee']), (5, ['l_ankle']), (6, ['pelvis']),
    (7, ['thorax']), (8, ['upper_neck']), (9, ['head top']), (10, ['r_wrist']),
    (11, ['r_elbow']), (12, ['r_shoulder']), (13, ['l_shoulder']),
    (14, ['l_elbow']), (15, ['l_wrist'])]

class MpiiExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__()

        self._categories = { AnnotationType.points:
            PointsCategories.from_iterable(MpiiPointsCategories) }

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        items = {}

        root_dir = osp.dirname(path)

        hb_path = osp.join(root_dir, MpiiPath.HEADBOXES_FILE)
        if osp.isfile(hb_path):
            headboxes = np.load(hb_path)
        else:
            headboxes = []

        vis_path = osp.join(root_dir, MpiiPath.VISIBILITY_FILE)
        if osp.isfile(vis_path):
            c = np.load(vis_path)
            visibility = np.load(vis_path).T
        else:
            visibility = []

        pos_gt_path = osp.join(root_dir, MpiiPath.POS_GT_FILE)
        if osp.isfile(pos_gt_path):
            r = np.load(pos_gt_path)
            gt_pose = np.transpose(np.load(pos_gt_path), (2, 0, 1))
        else:
            gt_pose = []

        with open(path) as f:
            for i, ann in enumerate(json.load(f)):
                item_id = osp.splitext(ann.get('img_paths'))[0]

                center = ann.get('objpos')
                scale = float(ann.get('scale_provided'))

                if np.size(gt_pose):
                    points = gt_pose[i]
                    points = points.reshape(points.shape[0] * points.shape[1])

                    if np.size(visibility):
                        vis = visibility[i]
                    else:
                        vis = np.ones(len(points) // 2)
                else:
                    keypoints = np.array(ann.get('joint_self'))
                    keypoints = keypoints.reshape(keypoints.shape[0] * keypoints.shape[1])
                    points = [p for i, p in enumerate(keypoints) if i % 3 != 2]

                    vis = keypoints[2::3]
                    if np.size(visibility):
                        vis = visibility[i]

                vis = [int(val) for val in vis]

                annotations = [Points(points, vis, attributes={'center': center,
                    'scale': scale})]

                if np.size(headboxes):
                    bbox = headboxes[:, :, i]
                    annotations.append(Bbox(bbox[0][0], bbox[0][1],
                        bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]))


                items[item_id] = DatasetItem(id=item_id, subset=self._subset,
                    image=Image(path=osp.join(root_dir, ann.get('img_paths'))),
                    annotations=annotations)

        return items

class MpiiImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, '.json', 'mpii')

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(MpiiPath.ANNOTATION_FILE)
