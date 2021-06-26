
# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT
class PointCloudPath:
    DEFAULT_DIR = "ds0"
    ANNNOTATION_DIR = "ann"
    IMAGE_EXT = ".jpg"
    POINT_CLOUD_DIR = "pointcloud"
    WRITE_FILES = ["meta.json", "key_id_map.json"]
    BUILTIN_ATTRS = {"occluded", "frame", "label_id", "user", "createdAt", "updatedAt"}
