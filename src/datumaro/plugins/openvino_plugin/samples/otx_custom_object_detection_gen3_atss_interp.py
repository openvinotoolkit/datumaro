# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import cv2

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.samples.utils import (
    create_bboxes_with_rescaling,
    rescale_img_keeping_aspect_ratio,
)

__all__ = ["OTXATSSModelInterpreter"]


class OTXATSSModelInterpreter(IModelInterpreter):
    h_model = 736
    w_model = 992

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        output = rescale_img_keeping_aspect_ratio(img, self.h_model, self.w_model)

        # From BGR to RGB
        output.image = cv2.cvtColor(output.image, cv2.COLOR_BGR2RGB)
        # From HWC to CHW
        output.image = output.image.transpose(2, 0, 1)

        return output.image, output.scale

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        scale = info
        r_scale = 1 / scale
        return create_bboxes_with_rescaling(pred["boxes"], pred["labels"], r_scale)

    def get_categories(self):
        return None
