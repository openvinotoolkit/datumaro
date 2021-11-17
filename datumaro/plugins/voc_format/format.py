# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from enum import Enum, auto
from itertools import chain

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, LabelCategories, MaskCategories,
)


class VocTask(Enum):
    classification = auto()
    detection = auto()
    segmentation = auto()
    action_classification = auto()
    person_layout = auto()

class VocLabel(Enum):
    background = 0
    aeroplane = 1
    bicycle = 2
    bird = 3
    boat = 4
    bottle = 5
    bus = 6
    car = 7
    cat = 8
    chair = 9
    cow = 10
    diningtable = 11
    dog = 12
    horse = 13
    motorbike = 14
    person = 15
    pottedplant = 16
    sheep = 17
    sofa = 18
    train = 19
    tvmonitor = 20
    ignored = 255

class VocPose(Enum):
    Unspecified = auto()
    Left = auto()
    Right = auto()
    Frontal = auto()
    Rear = auto()

class VocBodyPart(Enum):
    head = auto()
    hand = auto()
    foot = auto()

class VocAction(Enum):
    other = auto()
    jumping = auto()
    phoning = auto()
    playinginstrument = auto()
    reading = auto()
    ridingbike = auto()
    ridinghorse = auto()
    running = auto()
    takingphoto = auto()
    usingcomputer = auto()
    walking = auto()

def generate_colormap(length=256):
    def get_bit(number, index):
        return (number >> index) & 1

    colormap = np.zeros((length, 3), dtype=int)
    indices = np.arange(length, dtype=int)

    for j in range(7, -1, -1):
        for c in range(3):
            colormap[:, c] |= get_bit(indices, c) << j
        indices >>= 3

    return OrderedDict(
        (id, tuple(color)) for id, color in enumerate(colormap)
    )

VocColormap = {id: color for id, color in generate_colormap(256).items()
    if id in [l.value for l in VocLabel]}
VocInstColormap = generate_colormap(256)

class VocPath:
    IMAGES_DIR = 'JPEGImages'
    ANNOTATIONS_DIR = 'Annotations'
    SEGMENTATION_DIR = 'SegmentationClass'
    INSTANCES_DIR = 'SegmentationObject'
    SUBSETS_DIR = 'ImageSets'
    IMAGE_EXT = '.jpg'
    SEGM_EXT = '.png'
    LABELMAP_FILE = 'labelmap.txt'

    TASK_DIR = {
        VocTask.classification: 'Main',
        VocTask.detection: 'Main',
        VocTask.segmentation: 'Segmentation',
        VocTask.action_classification: 'Action',
        VocTask.person_layout: 'Layout',
    }


def make_voc_label_map():
    labels = sorted(VocLabel, key=lambda l: l.value)
    label_map = OrderedDict(
        (label.name, [VocColormap[label.value], [], []]) for label in labels)
    label_map[VocLabel.person.name][1] = [p.name for p in VocBodyPart]
    label_map[VocLabel.person.name][2] = [a.name for a in VocAction]
    return label_map

def make_voc_categories(label_map=None):
    if label_map is None:
        label_map = make_voc_label_map()

    categories = {}

    label_categories = LabelCategories()
    label_categories.attributes.update(['difficult', 'truncated', 'occluded'])

    for label, desc in label_map.items():
        label_categories.add(label, attributes=desc[2])
    for part in OrderedDict((k, None) for k in chain(
            *(desc[1] for desc in label_map.values()))):
        label_categories.add(part)
    categories[AnnotationType.label] = label_categories

    has_colors = any(v[0] is not None for v in label_map.values())
    if not has_colors: # generate new colors
        colormap = generate_colormap(len(label_map))
    else: # only copy defined colors
        label_id = lambda label: label_categories.find(label)[0]
        colormap = { label_id(name): desc[0]
            for name, desc in label_map.items() if desc[0] is not None }
    mask_categories = MaskCategories(colormap)
    mask_categories.inverse_colormap # pylint: disable=pointless-statement
    categories[AnnotationType.mask] = mask_categories

    return categories
