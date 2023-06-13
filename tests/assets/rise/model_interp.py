# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.abstracts.model_interpreter import IModelInterpreter
from datumaro.components.annotation import Label
from datumaro.util.annotation_util import softmax


class SsdMobilenetCocoDetectionModelInterpreter(IModelInterpreter):
    def normalize(self, inputs):
        return inputs

    def process_outputs(self, inputs, outputs):
        # inputs = model input; array or images; shape = (B, H, W, C)
        # outputs = model output; shape = (B, 3);
        # results = conversion result;
        # [B x [a score for label0, a score for label1, a score for label2]];

        return [
            [
                Label(label=label, attributes={"score": score})
                for label, score in enumerate(softmax(output))
            ]
            for output in outputs
        ]

    def get_categories(self):
        return None

    def resize(self, inputs):
        return inputs
