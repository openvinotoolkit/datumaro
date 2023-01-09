# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.annotation import AnnotationType, HashKey, LabelCategories


def process_outputs(inputs, outputs):
    results = [[HashKey(outputs, label=None)]]
    return results


def get_categories():
    label_categories = LabelCategories()
    return {AnnotationType.label: label_categories}
