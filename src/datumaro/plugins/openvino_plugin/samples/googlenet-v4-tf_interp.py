# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import cv2

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    FeatureVector,
    Label,
    LabelCategories,
)
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image


class GooglenetV4TfModelInterpreter(IModelInterpreter):
    LOGIT_KEY = "InceptionV4/Logits/Predictions"
    FEAT_KEY = "InceptionV4/Logits/PreLogitsFlatten/flatten_1/Reshape:0"

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        img = cv2.resize(img, (299, 299))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, None

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        logit = pred.get(self.LOGIT_KEY)
        if logit is None:
            raise DatumaroError(f'"{self.LOGIT_KEY}" key should exist in the model prediction.')

        feature_vector = pred.get(self.FEAT_KEY)
        if feature_vector is None:
            raise DatumaroError(f'"{self.FEAT_KEY}" key should exist in the model prediction.')

        outputs = [
            Label(label=label, attributes={"score": score}) for label, score in enumerate(logit)
        ]
        outputs += [FeatureVector(feature_vector)]

        return outputs  # [FeatureVector(logit), FeatureVector(feature_vector)]

    def get_categories(self):
        # output categories - label map etc.

        label_categories = LabelCategories()

        with open("samples/imagenet.class", "r", encoding="utf-8") as file:
            for line in file.readlines():
                label = line.strip()
                label_categories.add(label)

        return {AnnotationType.label: label_categories}
