# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from enum import IntEnum


class SynthiaFormatType(IntEnum):
    synthia_rand = 0
    synthia_sf = 1
    synthia_al = 2


class SynthiaRandPath:
    IMAGES_DIR = "RGB"
    LABELS_SEGM_DIR = "GTTXT"
    SEMANTIC_SEGM_DIR = "GT"

    @classmethod
    def meta_folders(cls):
        return [val for val in vars(cls).values() if isinstance(val, str) and "datumaro" not in val]


class SynthiaSfPath:
    IMAGES_DIR = "RGBLeft"
    SEMANTIC_SEGM_DIR = "GTLeft"
    DEPTH_DIR = "DepthLeft"

    @classmethod
    def meta_folders(cls):
        return [val for val in vars(cls).values() if isinstance(val, str) and "datumaro" not in val]


class SynthiaAlPath:
    IMAGES_DIR = "RGB"
    SEMANTIC_SEGM_DIR = "SemSeg"
    DEPTH_DIR = "Depth"

    @classmethod
    def meta_folders(cls):
        return [val for val in vars(cls).values() if isinstance(val, str) and "datumaro" not in val]


SynthiaRandLabelMap = OrderedDict(
    [
        ("Void", (0, 0, 0)),
        ("Sky", (128, 128, 128)),
        ("Building", (128, 0, 0)),
        ("Road", (128, 64, 128)),
        ("Sidewalk", (0, 0, 192)),
        ("Fence", (64, 64, 128)),
        ("Vegetation", (128, 128, 0)),
        ("Pole", (192, 192, 128)),
        ("Car", (64, 0, 128)),
        ("Sign", (192, 128, 128)),
        ("Pedestrian", (64, 64, 0)),
        ("Cyclist", (0, 128, 192)),
        ("Lanemarking", (0, 175, 0)),
        ("TrafficLight", (0, 128, 128)),
    ]
)


SynthiaSfLabelMap = OrderedDict(
    [
        ("Void", (0, 0, 0)),
        ("Road", (128, 64, 128)),
        ("Sidewalk", (244, 35, 232)),
        ("Building", (70, 70, 70)),
        ("Wall", (102, 102, 156)),
        ("Fence", (190, 153, 153)),
        ("Pole", (153, 153, 153)),
        ("TrafficLight", (250, 170, 30)),
        ("TrafficSign", (220, 220, 0)),
        ("Vegetation", (107, 142, 35)),
        ("Terrian", (152, 251, 152)),
        ("Sky", (70, 130, 180)),
        ("Person", (220, 20, 60)),
        ("Rider", (255, 0, 0)),
        ("Car", (0, 0, 142)),
        ("Truck", (0, 80, 100)),
        ("Bus", (0, 60, 100)),
        ("Train", (0, 80, 100)),
        ("Motorcycle", (0, 0, 230)),
        ("Bicycle", (119, 11, 31)),
        ("RoadLines", (157, 234, 50)),
        ("Other", (72, 0, 98)),
        ("RoadWorks", (167, 106, 29)),
    ]
)


SynthiaAlLabelMap = OrderedDict(
    [
        ("Void", (0, 0, 0)),
        ("Sky", (128, 128, 128)),
        ("Building", (128, 0, 0)),
        ("Road", (128, 64, 128)),
        ("Sidewalk", (0, 0, 192)),
        ("Fence", (64, 64, 128)),
        ("Vegetation", (128, 128, 0)),
        ("Pole", (192, 192, 128)),
        ("Car", (64, 0, 128)),
        ("TrafficSign", (192, 128, 128)),
        ("Pedestrian", (64, 64, 0)),
        ("Bicycle", (0, 128, 192)),
        ("Lanemarking", (0, 172, 0)),
        ("TrafficLight", (0, 128, 128)),
    ]
)
