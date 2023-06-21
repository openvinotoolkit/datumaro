# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import cv2
import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, AnnotationType, Bbox, LabelCategories


class FaceDetectionModelInterpreter(IModelInterpreter):
    height = 400
    width = 600
    conf_threshold = 0.5

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, PrepInfo]:
        assert img.ndim == 3
        h_img, w_img, _ = img.shape

        # This should be stored to reconstruct the bbox points
        img_size = h_img, w_img

        img = cv2.resize(img, (self.width, self.height))
        img = img.transpose(2, 0, 1).astype(np.dtype("<f"))
        return img, img_size

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        detection = pred.get("detection_out")
        h_img, w_img = info  # Info has the image size

        bboxes = []
        # detection has (1, 200, 7) dimensions (200 detections for an image)
        for idx, det in enumerate(detection[0, :, :]):
            conf = det[2]
            if conf <= self.conf_threshold:
                continue

            label = det[1]
            # These values are in [0, 1], so that we have to rescale them.
            x_min, y_min = w_img * det[3], h_img * det[4]
            x_max, y_max = w_img * det[5], h_img * det[6]
            bboxes.append(
                Bbox(
                    x=x_min,
                    y=y_min,
                    w=x_max - x_min,
                    h=y_max - y_min,
                    id=idx,
                    label=label,
                )
            )
        return bboxes

    def get_categories(self):
        label_categories = LabelCategories()
        label_categories.add(name="empty")
        label_categories.add(name="face")
        return {AnnotationType.label: label_categories}
