# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable, Optional

import numpy as np

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset


class FrameworkConverterFactory:
    @staticmethod
    def create_converter(framework):
        if framework == "torch":
            try:
                return DmTorchDataset
            except ImportError:
                raise ImportError("PyTorch package not found. Cannot convert to PyTorch dataset.")
        elif framework == "tf":
            try:
                return DmTfDataset
            except ImportError:
                raise ImportError(
                    "Tensorflow package not found. Cannot convert to Tensorflow dataset."
                )
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

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            dataitem = self.dataset.get(id=self._ids[idx], subset=self.subset)
            image = dataitem.media.data
            if image.dtype == np.uint8 or image.max() > 1:
                image = image.astype(np.float32) / 255

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            label = []
            for ann in dataitem.annotations:
                if self.task == "classification" and ann.type == AnnotationType.label:
                    label = ann.label
                    break
                elif self.task == "detection" and ann.type == AnnotationType.bbox:
                    label.append(ann.as_dict())
                elif self.task == "segmentation" and ann.type == AnnotationType.polygon:
                    label.append(ann.as_dict())
                elif self.task == "segmentation" and ann.type == AnnotationType.mask:
                    label = ann.image
                    break

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

                if self.task in ["detection", "segmentation"] and isinstance(
                    self.output_signature[1], dict
                ):
                    label = {}
                    for key, spec in self.output_signature[1].items():
                        label[key] = tf.convert_to_tensor(
                            [ann.as_dict().get(spec.name, None) for ann in item.annotations]
                        )
                elif self.task == "classification":
                    label = item.annotations[0].label
                elif self.task == "segmentation":
                    label = item.annotations[0].image

                yield image, label

        def create_tf_dataset(self):
            tf_dataset = tf.data.Dataset.from_generator(
                self.generator_wrapper, output_signature=self.output_signature
            )
            return tf_dataset

        def repeat(self, count=None):
            return self.create_tf_dataset().repeat(count)

        def batch(self, batch_size, drop_remainder=False):
            return self.create_tf_dataset().batch(batch_size, drop_remainder=drop_remainder)

except ImportError:

    class DmTfDataset:
        def __init__(self):
            raise ImportError("Tensorflow package not found. Cannot convert to Tensorflow dataset.")
