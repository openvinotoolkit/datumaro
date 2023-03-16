# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.annotation import AnnotationType, HashKey, LabelCategories


def normalize(inputs):
    return inputs


def process_outputs(inputs, outputs):
    results = [[HashKey(outputs)]]
    return results


def get_categories():
    label_categories = LabelCategories()
    return {AnnotationType.label: label_categories}
