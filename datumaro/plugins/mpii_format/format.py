# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

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
