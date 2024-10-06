# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np

from datumaro.components.abstracts import IModelInterpreter
from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image
from datumaro.plugins.openvino_plugin.samples.utils import gen_hash_key
from datumaro.util.samples import get_samples_path


class ClipViTL14ModelInterpreter(IModelInterpreter):
    mean = (255 * np.array([0.485, 0.456, 0.406])).reshape(1, 1, 3)
    std = (255 * np.array([0.229, 0.224, 0.225])).reshape(1, 1, 3)

    def preprocess(self, inp: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        img = inp.media_as(Image).data
        img = cv2.resize(img, (336, 336))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std

        if img.ndim == 3 and img.shape[2] in {3, 4}:
            img = np.transpose(img, (2, 0, 1))
        return img, None

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        feature_vector = pred.get("output")
        if feature_vector is None:
            raise DatumaroError('"output" key should exist in the model prediction.')

        return [gen_hash_key(feature_vector)]

    def get_categories(self):
        label_categories = LabelCategories()
        openvino_plugin_samples_dir = get_samples_path()
        imagenet_class_path = osp.join(openvino_plugin_samples_dir, "imagenet.class")

        with open(imagenet_class_path, "r", encoding="utf-8") as file:
            labels = [line.strip() for line in file]
            for label in labels:
                label_categories.add(label)

        return {AnnotationType.label: label_categories}
