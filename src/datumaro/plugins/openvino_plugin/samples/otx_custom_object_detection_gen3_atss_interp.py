# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import cv2
import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, Bbox

__all__ = ["OTXAtssModelInterpreter"]


class OTXAtssModelInterpreter(IModelInterpreter):
    h_model = 736
    w_model = 992

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, PrepInfo]:
        assert len(img.shape) == 3

        h_img, w_img = img.shape[:2]

        scale = min(self.h_model / h_img, self.w_model / w_img)

        h_resize = min(int(scale * h_img), self.h_model)
        w_resize = min(int(scale * w_img), self.w_model)

        num_channel = img.shape[-1]

        resized_inputs = np.zeros((self.h_model, self.w_model, num_channel), dtype=np.uint8)

        resized_inputs[:h_resize, :w_resize, :] = cv2.resize(
            img,
            (w_resize, h_resize),
            interpolation=cv2.INTER_LINEAR,
        )
        # From BGR to RGB
        resized_inputs = cv2.cvtColor(resized_inputs, cv2.COLOR_BGR2RGB)
        # From HWC to CHW
        resized_inputs = resized_inputs.transpose(2, 0, 1)

        return resized_inputs, scale

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        scale = info
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

        return _create_anns(pred["boxes"], pred["labels"])

    def get_categories(self):
        return None
