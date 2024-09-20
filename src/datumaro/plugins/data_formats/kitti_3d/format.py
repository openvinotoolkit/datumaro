# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from enum import Enum


class Kitti3dPath:
    PCD_DIR = osp.join("velodyne_points", "data")
    IMAGE_DIR = "image_2"
    LABEL_DIR = "label_2"
    CALIB_DIR = "calib"
    BUILTIN_ATTRS = {"occluded", "truncation", "occlusion"}
    SPECIAL_ATTRS = {
        "track_id",
    }
    ANNO_FILE = "tracklet_labels.xml"
    NAME_MAPPING_FILE = "frame_list.txt"


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
