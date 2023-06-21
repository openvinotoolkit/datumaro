# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import inspect
import logging as log
import os.path as osp
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from datumaro.components.abstracts.model_interpreter import IModelInterpreter, ModelPred, PrepInfo
from datumaro.components.annotation import Annotation
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.errors import DatumaroError


# pylint: disable=no-self-use
class Launcher(CliPlugin):
    def __init__(self, model_dir: Optional[str] = None):
        pass

    def preprocess(self, img: Image) -> Tuple[np.ndarray, PrepInfo]:
        """Preprocess single image before launch()

        Datumaro passes image data as `np.ndarray` with BGR format (H, W, C).
        The output should be also `np.ndarray` but it can be stacked into a batch of images.
        In this step, you usually implement resizing, normalizing, or color channel conversion
        for your launcher (or model).
        """
        raise NotImplementedError()

    def infer(self, inputs: np.ndarray) -> List[ModelPred]:
        """
        It executes the actual model inference for the inputs.

        Parameters:
            inputs: Batch of the numpy formatted input data
        Returns:
            List of model outputs
        """
        raise NotImplementedError()

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        raise NotImplementedError()

    def launch(self, batch: Sequence[DatasetItem]) -> List[List[Annotation]]:
        """Launch to obtain the inference outputs of items.

        Parameters:
            inputs: batch of Datasetitems

        Returns:
            A list of annotation list. Each annotation list is mapped to the input
            :class:`DatasetItem`. These annotation list are pseudo-labels obtained by
            the model inference.
        """
        if len(batch) == 0:
            return []

        inputs_img = []
        inputs_info = []

        for item in batch:
            prep_img, prep_info = self.preprocess(item.media)
            inputs_img.append(prep_img)
            inputs_info.append(prep_info)

        preds = self.infer(
            np.stack(
                [np.atleast_3d(img) for img in inputs_img],
                axis=0,
            )
        )

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

    def preprocess(self, img: Image) -> Tuple[np.ndarray, PrepInfo]:
        return self._interpreter.preprocess(img.data)

    def postprocess(self, pred: ModelPred, info: PrepInfo) -> List[Annotation]:
        return self._interpreter.postprocess(pred, info)

    def _load_interpreter(self, file_path: str) -> IModelInterpreter:
        fname, _ = osp.splitext(osp.basename(file_path))
        spec = spec_from_file_location(fname, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, IModelInterpreter)
                and obj is not IModelInterpreter
            ):
                log.info(f"Load {name} for model interpreter.")
                return obj()

        raise DatumaroError(f"{file_path} has no class derived from IModelInterpreter.")

    def categories(self):
        return self._interpreter.get_categories()
