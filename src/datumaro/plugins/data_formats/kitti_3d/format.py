# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import AnnotationType, LabelCategories


class Kitti3dPath:
    PCD_DIR = osp.join("velodyne")
    IMAGE_DIR = "image_2"
    LABEL_DIR = "label_2"
    CALIB_DIR = "calib"


Kitti3DLabelMap = [
    "DontCare",
    "Car",
    "Pedestrian",
    "Van",
    "Truck",
    "Cyclist",
    "Sitter",
    "Train",
    "Motorcycle",
    "Bus",
    "Misc",
]


def make_kitti3d_categories(label_map=None):
    if label_map is None:
        label_map = Kitti3DLabelMap

    categories = {}
    common_attrs = {"truncated", "occluded", "alpha", "dimensions", "location", "rotation_y"}
    label_categories = LabelCategories(attributes=common_attrs)
    for label in label_map:
        label_categories.add(label)
    categories[AnnotationType.label] = label_categories

    return categories
