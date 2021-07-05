# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum
import os.path as osp


class KittiRawPath:
    PCD_DIR = osp.join('velodyne_points', 'data')
    IMG_DIR_PREFIX = 'image_'
    BUILTIN_ATTRS = {'occluded'}

class PoseStates(Enum):
    UNSET = 0
    INTERP = 1
    LABELED = 2

class OcclusionStates(Enum):
    OCCLUSION_UNSET = -1
    VISIBLE = 0
    PARTLY = 1
    FULLY = 2

class TruncationStates(Enum):
    TRUNCATION_UNSET = -1
    IN_IMAGE = 0
    TRUNCATED = 1
    OUT_IMAGE = 2
    BEHIND_IMAGE = 99
