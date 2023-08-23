# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset

TASK_ANN_TYPE = {
    "classification": AnnotationType.label,
    "detection": AnnotationType.bbox,
    "instance_segmentation": AnnotationType.polygon,
    "semantic_segmentation": AnnotationType.mask,
}


class FrameworkConverterFactory:
    @staticmethod
    def create_converter(framework):
        if framework == "torch":
            return DmTorchDataset
        elif framework == "tf":
            return DmTfDataset
        else:
            raise ValueError("Unsupported framework")


class FrameworkConverter:
    def __init__(self, dataset, subset, task):
        self._dataset = dataset
        self._subset = subset
        self._task = task

    def to_framework(self, framework, **kwargs):
        converter_cls = FrameworkConverterFactory.create_converter(framework)
        return converter_cls(
            dataset=self._dataset,
            subset=self._subset,
            task=self._task,
            **kwargs,
        )


try:
    import torch

    class DmTorchDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            dataset: Dataset,
            subset: str,
            task: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
        ):
            self.dataset = dataset.get_subset(subset)
            self.subset = subset
            self.task = task
            self.transform = transform
            self.target_transform = target_transform
            self._ids = []
            for item in self.dataset:
                self._ids.append(item.id)

        def __len__(self) -> int:
            return len(self.dataset)

        def __getitem__(self, idx):
            dataitem = self.dataset.get(id=self._ids[idx], subset=self.subset)
            image = dataitem.media.data
            if image.dtype == np.uint8 or image.max() > 1:
                image = image.astype(np.float32) / 255

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            if self.task == "classification":
                label = [
                    ann.label
                    for ann in dataitem.annotations
                    if ann.type == TASK_ANN_TYPE[self.task]
                ][0]
            elif self.task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in dataitem.annotations
                    if ann.type == TASK_ANN_TYPE[self.task]
                ]
            elif self.task == "semantic_segmentation":
                label = [
                    ann.image
                    for ann in dataitem.annotations
                    if ann.type == TASK_ANN_TYPE[self.task]
                ][0]

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                label = self.target_transform(label)

            return image, label

except ImportError:

    class DmTorchDataset:
        def __init__(self):
            raise ImportError("PyTorch package not found. Cannot convert to PyTorch dataset.")


try:
    import tensorflow as tf

    class DmTfDataset:
        def __init__(
            self,
            dataset: Dataset,
            subset: str,
            task: str,
            output_signature: Optional[tuple] = None,
        ):
            self.dataset = dataset.get_subset(subset)
            self.task = task
            self.output_signature = output_signature

        def generator_wrapper(self):
            for item in self.dataset:
                image = item.media.data

                if self.task == "classification":
                    label = [
                        ann.label
                        for ann in item.annotations
                        if ann.type == TASK_ANN_TYPE[self.task]
                    ][0]
                elif self.task in ["detection", "instance_segmentation"] and isinstance(
                    self.output_signature[1], dict
                ):
                    label = {}
                    for key, spec in self.output_signature[1].items():
                        label[key] = tf.convert_to_tensor(
                            [
                                ann.as_dict().get(spec.name, None)
                                for ann in item.annotations
                                if ann.type == TASK_ANN_TYPE[self.task]
                            ]
                        )
                elif self.task == "semantic_segmentation":
                    label = [
                        ann.image
                        for ann in item.annotations
                        if ann.type == TASK_ANN_TYPE[self.task]
                    ][0]

                yield image, label

        def create(self) -> tf.data.Dataset:
            tf_dataset = tf.data.Dataset.from_generator(
                self.generator_wrapper, output_signature=self.output_signature
            )
            return tf_dataset

        def repeat(self, count=None) -> tf.data.Dataset:
            return self.create().repeat(count)

        def batch(self, batch_size, drop_remainder=False) -> tf.data.Dataset:
            return self.create().batch(batch_size, drop_remainder=drop_remainder)

except ImportError:

    class DmTfDataset:
        def __init__(self):
            raise ImportError("Tensorflow package not found. Cannot convert to Tensorflow dataset.")
