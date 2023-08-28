# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
import logging as log
import os.path as osp
from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from typing import Dict, List, Optional, Sequence, Tuple, overload

import numpy as np

from datumaro.components.abstracts.model_interpreter import (
    IModelInterpreter,
    LauncherInputType,
    ModelPred,
    PrepInfo,
)
from datumaro.components.annotation import Annotation
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.errors import DatumaroError


# pylint: disable=no-self-use
class Launcher(CliPlugin):
    def __init__(self, model_dir: Optional[str] = None):
        pass

    def preprocess(self, item: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        """Preprocess single dataset item before launch()

        There are two output types:

        1. The output is `np.ndarray`. For example, it can be image data as `np.ndarray` with BGR format (H, W, C).
        In this step, you usually implement resizing, normalizing, or color channel conversion
        for your launcher (or model).

        2. The output is `Dict[str, np.ndarray]`. For example, it can be image and text pairs.
        Therefore, this can be used for the model having multi modality for image and text inputs.
        """
        raise NotImplementedError()

    @overload
    def infer(self, inputs: Dict[str, np.ndarray]) -> List[ModelPred]:
        """
        It executes the actual model inference for the inputs.

        Parameters:
            inputs: Dictionary of input data (its key is string and value is numpy array).
        Returns:
            List of model outputs
        """
        ...

    @overload
    def infer(self, inputs: np.ndarray) -> List[ModelPred]:
        """
        It executes the actual model inference for the inputs.

        Parameters:
            inputs: Batch of the numpy formatted input data
        Returns:
            List of model outputs
        """
        ...

    def infer(self, inputs: LauncherInputType) -> List[ModelPred]:
        raise NotImplementedError

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        raise NotImplementedError()

    def launch(self, batch: Sequence[DatasetItem], stack: bool = True) -> List[List[Annotation]]:
        """Launch to obtain the inference outputs of items.

        Parameters:
            inputs: batch of Datasetitems
            stack: If true, launch inference for the stacked input for the batch-wise dimension
                Otherwise, launch inference for each input.

        Returns:
            A list of annotation list. Each annotation list is mapped to the input
            :class:`DatasetItem`. These annotation list are pseudo-labels obtained by
            the model inference.
        """
        if len(batch) == 0:
            return []

        inputs = []
        inputs_info = []

        for item in batch:
            prep_inp, prep_info = self.preprocess(item)
            inputs.append(prep_inp)
            inputs_info.append(prep_info)

        if stack:
            if all(isinstance(inp, np.ndarray) for inp in inputs):
                preds = self.infer(np.stack(inputs, axis=0))
            elif all(isinstance(inp, dict) for inp in inputs):
                grouped_by_key = defaultdict(list)
                for inp in inputs:
                    for k, v in inp.items():
                        grouped_by_key[k].append(v)

                grouped_inputs = {k: np.stack(v, axis=0) for k, v in grouped_by_key.items()}
                preds = self.infer(grouped_inputs)
            else:
                actual = [type(inp) for inp in inputs]
                raise DatumaroError(f"Inputs should be np.ndarray or dict. but {actual}")
        else:
            preds = [self.infer(inp) for inp in inputs]

        return [self.postprocess(pred, info) for pred, info in zip(preds, inputs_info)]

    def infos(self):
        return None

    def categories(self):
        return None

    def type_check(self, item: DatasetItem) -> bool:
        """Check the media type of dataset item.

        If False, the item is excluded from the input batch.
        """
        return True


class LauncherWithModelInterpreter(Launcher):
    def __init__(self, model_interpreter_path: str):
        self._interpreter = self._load_interpreter(file_path=model_interpreter_path)

    def preprocess(self, item: DatasetItem) -> Tuple[LauncherInputType, PrepInfo]:
        """Preprocessing an input DatasetItem.

        Parameters:
            img: Input Datasetitem

        Returns:
            It returns a tuple of preprocessed input and preprocessing information.
            The preprocessing information will be used in the postprocessing step.
            One use case for this would be an affine transformation of the output bounding box
            obtained by object detection models. Input images for those models are usually resized
            to fit the model input dimensions. As a result, the postprocessing step should refine
            the model output bounding box to match the original input image size.
        """
        return self._interpreter.preprocess(item)

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        """Postprocess a model prediction.

        Parameters:
            pred: Model prediction
            info: Preprocessing information coming from the preprocessing step

        Returns:
            A list of annotations which is created from the model predictions
        """
        return self._interpreter.postprocess(pred, info)

    def _load_interpreter(self, file_path: str) -> IModelInterpreter:
        fname, _ = osp.splitext(osp.basename(file_path))
        spec = spec_from_file_location(fname, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        interps = []
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, IModelInterpreter)
                and obj is not IModelInterpreter
            ):
                log.info(f"Load {name} for model interpreter.")
                interps.append(obj)

        if len(interps) == 0:
            raise DatumaroError(f"{file_path} has no class derived from IModelInterpreter.")

        if len(interps) > 1:
            raise DatumaroError(
                f"{file_path} has more than two classes derived from IModelInterpreter ({interps}). "
                "There should be only one ModelInterpreter in the file."
            )

        return interps.pop()()

    def categories(self):
        return self._interpreter.get_categories()
