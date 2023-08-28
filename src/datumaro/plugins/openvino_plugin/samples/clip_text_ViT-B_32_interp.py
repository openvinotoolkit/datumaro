# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.samples.utils import gen_hash_key


class ClipTextViTB32ModelInterpreter(IModelInterpreter):
    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        return img, None

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        feature_vector = pred.get("output")
        if feature_vector is None:
            raise DatumaroError('"output" key should exist in the model prediction.')

        return [gen_hash_key(feature_vector)]

    def get_categories(self):
        label_categories = LabelCategories()
        return {AnnotationType.label: label_categories}
