# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.annotation import AnnotationType, HashKey, LabelCategories


class ClipTextViTB32ModelInterpreter(IModelInterpreter):
    def normalize(self, inputs):
        return inputs

    def process_outputs(self, inputs, outputs):
        results = [[HashKey(outputs)]]
        return results

    def get_categories(self):
        label_categories = LabelCategories()
        return {AnnotationType.label: label_categories}

    def resize(self, inputs):
        return inputs
