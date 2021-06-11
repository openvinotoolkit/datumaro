
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

class VelodynePointsPath:
    IMAGES_DIR = "velodyne_points/data"
    RELATED_DIR = "image_"
    BUILTIN_ATTRS = set("occluded")


class VelodynePointsState:
    POSE_STATES = {"UNSET": 0, "INTERP": 1, "LABELED": 2}
    OCCLUSION_STATES = {"OCCLUSION_UNSET": -1, "VISIBLE": 0, "PARTLY": 1, "FULLY": 2}
    TRUNCATION_STATE = {"TRUNCATION_UNSET": -1, "IN_IMAGE": 0, "TRUNCATED": 1, "OUT_IMAGE": 2, "BEHIND_IMAGE": 99}
