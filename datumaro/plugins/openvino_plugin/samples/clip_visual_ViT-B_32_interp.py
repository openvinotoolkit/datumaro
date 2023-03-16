# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

from datumaro.components.annotation import AnnotationType, HashKey, LabelCategories


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
    return {AnnotationType.label: label_categories}
