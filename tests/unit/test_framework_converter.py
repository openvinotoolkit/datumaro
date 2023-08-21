# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging
import os
import os.path as osp
import pickle
import tempfile
from typing import List, Sequence  # nosec B403
from unittest import TestCase, mock, skipIf

import numpy as np
import pytest

from datumaro.components.dataset import DEFAULT_FORMAT, Dataset, eager_mode
from datumaro.plugins.framework_converter import MultiFrameworkDataset

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True


class FrameworkConverterTest(TestCase):
    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_torch_classification_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((64, 64)),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            torch_dataset = datasets.MNIST(
                root=tmp_dir,
                train=True,
                download=True,
                transform=transform,
            )

            dm_dataset = Dataset.import_from(path=osp.join(tmp_dir, "MNIST"), format="mnist")

            multi_framework_dataset = MultiFrameworkDataset(
                dm_dataset, subset="train", task="classification"
            )
            dm_torch_dataset = multi_framework_dataset.to_framework(
                framework="torch", transform=transform
            )

            for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
                assert torch.equal(torch_item[0], dm_item[0])
                self.assertEqual(torch_item[1], dm_item[1])

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_torch_detection_data(self):
        DUMMY_DATASET_DIR = get_test_asset_path("coco_dataset")
        format = "coco_instances"

        data_path = osp.join(DUMMY_DATASET_DIR, format)

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
            ]
        )

        torch_dataset = datasets.CocoDetection(
            root=osp.join(data_path, "images/train/"),
            annFile=osp.join(data_path, "annotations/instances_train.json"),
            transform=transform,
        )

        dm_dataset = Dataset.import_from(data_path, format)

        multi_framework_dataset = MultiFrameworkDataset(
            dm_dataset, subset="train", task="detection"
        )
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", transform=transform
        )

        for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
            assert torch.equal(torch_item[0], dm_item[0])

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_tf_classification_data(self):
        output_signature = (
            tf.TensorSpec(shape=(28, 28), dtype=tf.uint8),  # Modify shape and dtype accordingly
            tf.TensorSpec(shape=(), dtype=tf.uint8),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            tf_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

            keras_data_dir = osp.expanduser("~/.keras/datasets")
            dm_dataset = Dataset.import_from(
                path=osp.join(keras_data_dir, "fashion-mnist"), format="mnist"
            )

            multi_framework_dataset = MultiFrameworkDataset(
                dm_dataset, subset="test", task="classification"
            )
            dm_tf_dataset = multi_framework_dataset.to_framework(
                framework="tf", output_signature=output_signature
            )

            epoch, batch_size = 1, 16
            for tf_item, dm_item in zip(
                tf_dataset.repeat(epoch).batch(batch_size),
                dm_tf_dataset.repeat(epoch).batch(batch_size),
            ):
                assert tf.reduce_all(tf_item[0] == dm_item[0])
                assert tf.reduce_all(tf_item[1] == dm_item[1])
