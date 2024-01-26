# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import numpy as np

from datumaro.components.abstracts.model_interpreter import (
    IModelInterpreter,
    LauncherInputType,
    ModelPred,
    PrepInfo,
)
from datumaro.components.annotation import Annotation, Label
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.util.annotation_util import softmax


class DummyModelInterpreter(IModelInterpreter):
    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        img = np.transpose(img, (2, 0, 1))
        return super().preprocess(img)

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        output = pred.get("2")
        if output is None:
            raise ValueError()

        return [
            Label(label=label, attributes={"score": score})
            for label, score in enumerate(softmax(output))
        ]

    def get_categories(self):
        return None
