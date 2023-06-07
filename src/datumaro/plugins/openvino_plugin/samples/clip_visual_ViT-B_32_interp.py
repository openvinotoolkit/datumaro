# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, HashKey, LabelCategories
from datumaro.util.samples import get_samples_path


def normalize(inputs):
    mean = 255 * np.array([0.485, 0.456, 0.406])
    std = 255 * np.array([0.229, 0.224, 0.225])

    normalized_inputs = np.empty_like(inputs, dtype=inputs.dtype)
    for k, inp in enumerate(inputs):
        normalized_inputs[k] = (inp - mean[:, None, None]) / std[:, None, None]
    inputs = normalized_inputs

    return inputs


def process_outputs(inputs, outputs):
    results = [[HashKey(outputs)]]
    return results


def get_categories():
    label_categories = LabelCategories()

    openvino_plugin_samples_dir = get_samples_path()
    imagenet_class_path = osp.join(openvino_plugin_samples_dir, "imagenet.class")
    with open(imagenet_class_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            label = line.strip()
            label_categories.add(label)

    return {AnnotationType.label: label_categories}
