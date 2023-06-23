# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TypeVar

import numpy as np

from datumaro.components.annotation import Annotation

__all__ = ["IModelInterpreter"]

PrepInfo = TypeVar("PrepInfo")
ModelPred = Dict[str, np.ndarray]


class IModelInterpreter(ABC):
    @abstractmethod
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, PrepInfo]:
        """Preprocessing an input image.

        Parameters:
            img: Input image

        Returns:
            It returns a tuple of preprocessed image and preprocessing information.
            The preprocessing information will be used in the postprocessing step.
            One use case for this would be an affine transformation of the output bounding box
            obtained by object detection models. Input images for those models are usually resized
            to fit the model input dimensions. As a result, the postprocessing step should refine
            the model output bounding box to match the original input image size.
        """
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        """Postprocess a model prediction.

        Parameters:
            pred: Model prediction
            info: Preprocessing information coming from the preprocessing step

        Returns:
            A list of annotations which is created from the model predictions
        """
        raise NotImplementedError("Function should be implemented.")

    @abstractmethod
    def get_categories(self):
        """It should be implemented if the model generate a new categories"""
        raise NotImplementedError("Function should be implemented.")
