# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from enum import Enum, auto

from datumaro.components.annotation import AnnotationType, LabelCategories, MaskCategories
from datumaro.util import parse_json_file
from datumaro.util.mask_tools import generate_colormap


def parse_config_file(config_path):
    label_map = OrderedDict()
    config = parse_json_file(config_path)
    for label in config["labels"]:
        label_map[label["name"]] = tuple(map(int, label["color"]))
    return label_map


def get_parent_label(label):
    parent = label.split("--")[0]
    if parent == label:
        parent = ""
    return parent


def make_mapillary_instance_categories(label_map):
    label_cat = LabelCategories()
    for label, _ in label_map.items():
        label_cat.add(label, get_parent_label(label))

    has_colors = any([len(color) == 3 for color in label_map.values()])
    if not has_colors:
        colormap = generate_colormap(len(label_map))
    else:
        colormap = {
            label_cat.find(label, get_parent_label(label))[0]: color
            for label, color in label_map.items()
        }

    mask_cat = MaskCategories(colormap)
    mask_cat.inverse_colormap  # pylint: disable=pointless-statement

    return {AnnotationType.label: label_cat, AnnotationType.mask: mask_cat}


class MapillaryVistasTask(Enum):
    instances = auto()
    panoptic = auto()


class MapillaryVistasPath:
    IMAGES_DIR = "images"
    INSTANCES_DIR = "instances"
    PANOPTIC_DIR = "panoptic"
    POLYGON_DIR = "polygons"
    CLASS_DIR = "labels"
    MASK_EXT = ".png"

    ANNOTATION_DIRS = {
        "v1.2": [CLASS_DIR, INSTANCES_DIR],
        "v2.0": [CLASS_DIR, INSTANCES_DIR, PANOPTIC_DIR, POLYGON_DIR],
    }

    CONFIG_FILES = {"v1.2": "config_v1.2.json", "v2.0": "config_v2.0.json"}
    PANOPTIC_CONFIG = "panoptic_2020.json"

    CLASS_BY_DIR = {
        INSTANCES_DIR: MapillaryVistasTask.instances,
        CLASS_DIR: MapillaryVistasTask.instances,
        POLYGON_DIR: MapillaryVistasTask.instances,
        PANOPTIC_DIR: MapillaryVistasTask.panoptic,
    }


MapillaryVistasLabelMap_V12 = OrderedDict(
    [
        ("animal--bird", (165, 42, 42)),
        ("animal--ground-animal", (0, 192, 0)),
        ("construction--barrier--curb", (196, 196, 196)),
        ("construction--barrier--fence", (190, 153, 153)),
        ("construction--barrier--guard-rail", (180, 165, 180)),
        ("construction--barrier--other-barrier", (90, 120, 150)),
        ("construction--barrier--wall", (102, 102, 156)),
        ("construction--flat--bike-lane", (128, 64, 255)),
        ("construction--flat--crosswalk-plain", (140, 140, 200)),
        ("construction--flat--curb-cut", (170, 170, 170)),
        ("construction--flat--parking", (250, 170, 160)),
        ("construction--flat--pedestrian-area", (96, 96, 96)),
        ("construction--flat--rail-track", (230, 150, 140)),
        ("construction--flat--road", (128, 64, 128)),
        ("construction--flat--service-lane", (110, 110, 110)),
        ("construction--flat--sidewalk", (244, 35, 232)),
        ("construction--structure--bridge", (150, 100, 100)),
        ("construction--structure--building", (70, 70, 70)),
        ("construction--structure--tunnel", (150, 120, 90)),
        ("human--person", (220, 20, 60)),
        ("human--rider--bicyclist", (255, 0, 0)),
        ("human--rider--motorcyclist", (255, 0, 100)),
        ("human--rider--other-rider", (255, 0, 200)),
        ("marking--crosswalk-zebra", (200, 128, 128)),
        ("marking--general", (255, 255, 255)),
        ("nature--mountain", (64, 170, 64)),
        ("nature--sand", (230, 160, 50)),
        ("nature--sky", (70, 130, 180)),
        ("nature--snow", (190, 255, 255)),
        ("nature--terrain", (152, 251, 152)),
        ("nature--vegetation", (107, 142, 35)),
        ("nature--water", (0, 170, 30)),
        ("object--banner", (255, 255, 128)),
        ("object--bench", (250, 0, 30)),
        ("object--bike-rack", (100, 140, 180)),
        ("object--billboard", (220, 220, 220)),
        ("object--catch-basin", (220, 128, 128)),
        ("object--cctv-camera", (222, 40, 40)),
        ("object--fire-hydrant", (100, 170, 30)),
        ("object--junction-box", (40, 40, 40)),
        ("object--mailbox", (33, 33, 33)),
        ("object--manhole", (100, 128, 160)),
        ("object--phone-booth", (142, 0, 0)),
        ("object--pothole", (70, 100, 150)),
        ("object--street-light", (210, 170, 100)),
        ("object--support--pole", (153, 153, 153)),
        ("object--support--traffic-sign-frame", (128, 128, 128)),
        ("object--support--utility-pole", (0, 0, 80)),
        ("object--traffic-light", (250, 170, 30)),
        ("object--traffic-sign--back", (192, 192, 192)),
        ("object--traffic-sign--front", (220, 220, 0)),
        ("object--trash-can", (140, 140, 20)),
        ("object--vehicle--bicycle", (119, 11, 32)),
        ("object--vehicle--boat", (150, 0, 255)),
        ("object--vehicle--bus", (0, 60, 100)),
        ("object--vehicle--car", (0, 0, 142)),
        ("object--vehicle--caravan", (0, 0, 90)),
        ("object--vehicle--motorcycle", (0, 0, 230)),
        ("object--vehicle--on-rails", (0, 80, 100)),
        ("object--vehicle--other-vehicle", (128, 64, 64)),
        ("object--vehicle--trailer", (0, 0, 110)),
        ("object--vehicle--truck", (0, 0, 70)),
        ("object--vehicle--wheeled-slow", (0, 0, 192)),
        ("void--car-mount", (32, 32, 32)),
        ("void--ego-vehicle", (120, 10, 10)),
        ("void--unlabeled", (0, 0, 0)),
    ]
)

MapillaryVistasLabelMap_V20 = OrderedDict(
    [
        ("animal--bird", (165, 42, 42)),
        ("animal--ground-animal", (0, 192, 0)),
        ("construction--barrier--ambiguous", (250, 170, 31)),
        ("construction--barrier--concrete-block", (250, 170, 32)),
        ("construction--barrier--curb", (196, 196, 196)),
        ("construction--barrier--fence", (190, 153, 153)),
        ("construction--barrier--guard-rail", (180, 165, 180)),
        ("construction--barrier--other-barrier", (90, 120, 150)),
        ("construction--barrier--road-median", (250, 170, 33)),
        ("construction--barrier--road-side", (250, 170, 34)),
        ("construction--barrier--separator", (128, 128, 128)),
        ("construction--barrier--temporary", (250, 170, 35)),
        ("construction--barrier--wall", (102, 102, 156)),
        ("construction--flat--bike-lane", (128, 64, 255)),
        ("construction--flat--crosswalk-plain", (140, 140, 200)),
        ("construction--flat--curb-cut", (170, 170, 170)),
        ("construction--flat--driveway", (250, 170, 36)),
        ("construction--flat--parking", (250, 170, 160)),
        ("construction--flat--parking-aisle", (250, 170, 37)),
        ("construction--flat--pedestrian-area", (96, 96, 96)),
        ("construction--flat--rail-track", (230, 150, 140)),
        ("construction--flat--road", (128, 64, 128)),
        ("construction--flat--road-shoulder", (110, 110, 110)),
        ("construction--flat--service-lane", (110, 110, 110)),
        ("construction--flat--sidewalk", (244, 35, 232)),
        ("construction--flat--traffic-island", (128, 196, 128)),
        ("construction--structure--bridge", (150, 100, 100)),
        ("construction--structure--building", (70, 70, 70)),
        ("construction--structure--garage", (150, 150, 150)),
        ("construction--structure--tunnel", (150, 120, 90)),
        ("human--person--individual", (220, 20, 60)),
        ("human--person--person-group", (220, 20, 60)),
        ("human--rider--bicyclist", (255, 0, 0)),
        ("human--rider--motorcyclist", (255, 0, 100)),
        ("human--rider--other-rider", (255, 0, 200)),
        ("marking--continuous--dashed", (255, 255, 255)),
        ("marking--continuous--solid", (255, 255, 255)),
        ("marking--continuous--zigzag", (250, 170, 29)),
        ("marking--discrete--ambiguous", (250, 170, 28)),
        ("marking--discrete--arrow--left", (250, 170, 26)),
        ("marking--discrete--arrow--other", (250, 170, 25)),
        ("marking--discrete--arrow--right", (250, 170, 24)),
        ("marking--discrete--arrow--split-left-or-straight", (250, 170, 22)),
        ("marking--discrete--arrow--split-right-or-straight", (250, 170, 21)),
        ("marking--discrete--arrow--straight", (250, 170, 20)),
        ("marking--discrete--crosswalk-zebra", (255, 255, 255)),
        ("marking--discrete--give-way-row", (250, 170, 19)),
        ("marking--discrete--give-way-single", (250, 170, 18)),
        ("marking--discrete--hatched--chevron", (250, 170, 12)),
        ("marking--discrete--hatched--diagonal", (250, 170, 11)),
        ("marking--discrete--other-marking", (255, 255, 255)),
        ("marking--discrete--stop-line", (255, 255, 255)),
        ("marking--discrete--symbol--bicycle", (250, 170, 16)),
        ("marking--discrete--symbol--other", (250, 170, 15)),
        ("marking--discrete--text", (250, 170, 15)),
        ("marking-only--continuous--dashed", (255, 255, 255)),
        ("marking-only--discrete--crosswalk-zebra", (255, 255, 255)),
        ("marking-only--discrete--other-marking", (255, 255, 255)),
        ("marking-only--discrete--text", (255, 255, 255)),
        ("nature--mountain", (64, 170, 64)),
        ("nature--sand", (230, 160, 50)),
        ("nature--sky", (70, 130, 180)),
        ("nature--snow", (190, 255, 255)),
        ("nature--terrain", (152, 251, 152)),
        ("nature--vegetation", (107, 142, 35)),
        ("nature--water", (0, 170, 30)),
        ("object--banner", (255, 255, 128)),
        ("object--bench", (250, 0, 30)),
        ("object--bike-rack", (100, 140, 180)),
        ("object--catch-basin", (220, 128, 128)),
        ("object--cctv-camera", (222, 40, 40)),
        ("object--fire-hydrant", (100, 170, 30)),
        ("object--junction-box", (40, 40, 40)),
        ("object--mailbox", (33, 33, 33)),
        ("object--manhole", (100, 128, 160)),
        ("object--parking-meter", (20, 20, 255)),
        ("object--phone-booth", (142, 0, 0)),
        ("object--pothole", (70, 100, 150)),
        ("object--sign--advertisement", (250, 171, 30)),
        ("object--sign--ambiguous", (250, 172, 30)),
        ("object--sign--back", (250, 173, 30)),
        ("object--sign--information", (250, 174, 30)),
        ("object--sign--other", (250, 175, 30)),
        ("object--sign--store", (250, 176, 30)),
        ("object--street-light", (210, 170, 100)),
        ("object--support--pole", (153, 153, 153)),
        ("object--support--pole-group", (153, 153, 153)),
        ("object--support--traffic-sign-frame", (128, 128, 128)),
        ("object--support--utility-pole", (0, 0, 80)),
        ("object--traffic-cone", (210, 60, 60)),
        ("object--traffic-light--cyclists", (250, 170, 30)),
        ("object--traffic-light--general-horizontal", (250, 170, 30)),
        ("object--traffic-light--general-single", (250, 170, 30)),
        ("object--traffic-light--general-upright", (250, 170, 30)),
        ("object--traffic-light--other", (250, 170, 30)),
        ("object--traffic-light--pedestrians", (250, 170, 30)),
        ("object--traffic-sign--ambiguous", (192, 192, 192)),
        ("object--traffic-sign--back", (192, 192, 192)),
        ("object--traffic-sign--direction-back", (192, 192, 192)),
        ("object--traffic-sign--direction-front", (220, 220, 0)),
        ("object--traffic-sign--front", (220, 220, 0)),
        ("object--traffic-sign--information-parking", (0, 0, 196)),
        ("object--traffic-sign--temporary-back", (192, 192, 192)),
        ("object--traffic-sign--temporary-front", (220, 220, 0)),
        ("object--trash-can", (140, 140, 20)),
        ("object--vehicle--bicycle", (119, 11, 32)),
        ("object--vehicle--boat", (150, 0, 255)),
        ("object--vehicle--bus", (0, 60, 100)),
        ("object--vehicle--car", (0, 0, 142)),
        ("object--vehicle--caravan", (0, 0, 90)),
        ("object--vehicle--motorcycle", (0, 0, 230)),
        ("object--vehicle--on-rails", (0, 80, 100)),
        ("object--vehicle--other-vehicle", (128, 64, 64)),
        ("object--vehicle--trailer", (0, 0, 110)),
        ("object--vehicle--truck", (0, 0, 70)),
        ("object--vehicle--vehicle-group", (0, 0, 142)),
        ("object--vehicle--wheeled-slow", (0, 0, 192)),
        ("object--water-valve", (170, 170, 170)),
        ("void--car-mount", (32, 32, 32)),
        ("void--dynamic", (111, 74, 0)),
        ("void--ego-vehicle", (120, 10, 10)),
        ("void--ground", (81, 0, 81)),
        ("void--static", (111, 111, 0)),
        ("void--unlabeled", (0, 0, 0)),
    ]
)

MapillaryVistasLabelMaps = {
    "v1.2": MapillaryVistasLabelMap_V12,
    "v2.0": MapillaryVistasLabelMap_V20,
}
