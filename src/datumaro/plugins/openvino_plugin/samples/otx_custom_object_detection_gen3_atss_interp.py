# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.annotation import Bbox

__all__ = ["OTXAtssModelInterpreter"]


class OTXAtssModelInterpreter(IModelInterpreter):
    h_model = 736
    w_model = 992

    def __init__(self) -> None:
        self._scales = []

    def get_categories(self):
        return None

    def process_outputs(self, inputs, outputs):
        scale = self._scales.pop()
        r_scale = 1 / scale

        def _create_anns(bboxes, labels):
            idx = 0
            anns = []
            for bbox, label in zip(bboxes, labels):
                points = r_scale * bbox[:4]
                x1, y1, x2, y2 = points
                conf = bbox[4]
                anns.append(
                    Bbox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        id=idx,
                        label=label,
                        attributes={"score": conf},
                    )
                )
                idx += 1
            return anns

        return [
            _create_anns(bboxes, labels)
            for bboxes, labels in zip(outputs["boxes"], outputs["labels"])
        ]

    def normalize(self, inputs):
        return inputs

    def resize(self, inputs: np.ndarray):
        assert len(inputs.shape) == 4

        h_img, w_img = inputs.shape[1:3]

        scale = min(self.h_model / h_img, self.w_model / w_img)

        h_resize = min(int(scale * h_img), self.h_model)
        w_resize = min(int(scale * w_img), self.w_model)

        batch_size = inputs.shape[0]
        num_channel = inputs.shape[-1]

        resized_inputs = np.zeros(
            (batch_size, self.h_model, self.w_model, num_channel), dtype=np.uint8
        )

        for i in range(batch_size):
            resized_inputs[i, :h_resize, :w_resize, :] = cv2.resize(
                inputs[i],
                (w_resize, h_resize),
                interpolation=cv2.INTER_LINEAR,
            )

        self._scales += [scale]
        return resized_inputs
