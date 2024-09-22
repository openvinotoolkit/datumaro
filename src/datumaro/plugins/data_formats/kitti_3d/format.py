# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp


class Kitti3dPath:
    PCD_DIR = osp.join("velodyne")
    IMAGE_DIR = "image_2"
    LABEL_DIR = "label_2"
    CALIB_DIR = "calib"
