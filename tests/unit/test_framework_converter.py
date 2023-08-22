# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from unittest import TestCase, skipIf
from unittest.mock import Mock

import numpy as np

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, Label, Mask, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.framework_converter import (
    DmTfDataset,
    DmTorchDataset,
    FrameworkConverter,
    FrameworkConverterFactory,
)

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
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True


class FrameworkConverterTest(TestCase):
    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_create_converter_torch(self):
        converter = FrameworkConverterFactory.create_converter("torch")
        self.assertEqual(converter, DmTorchDataset)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_create_converter_tf(self):
        converter = FrameworkConverterFactory.create_converter("tf")
        self.assertEqual(converter, DmTfDataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_create_converter_invalid(self):
        with self.assertRaises(ValueError):
            FrameworkConverterFactory.create_converter("invalid_framework")


class TorchConverterTest(TestCase):
    def setUp(self):
        self.mock_dataset = Mock(spec=Dataset)
        self.mock_annotation = Mock(spec=Annotation)
        self.mock_media_item = Mock(spec=DatasetItem)
        self.mock_media_item.id = "0"
        self.mock_media_item.subset = "subset"
        self.mock_media_item.media = Image.from_numpy(np.array([[1, 2], [3, 4]]))
        self.mock_media_item.annotations = [self.mock_annotation]
        self.mock_dataset = Dataset.from_iterable([self.mock_media_item])
        self.transform = transforms.ToTensor()

    # @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    # def test_torch_dataset_import(self):
    #     with self.assertRaises(ImportError):
    #         from datumaro.plugins.framework_converter import DmTorchDataset

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_torch_dataset_len(self):
        dm_torch_dataset = DmTorchDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="classification",
            transform=self.transform,
        )
        length = len(dm_torch_dataset)

        self.assertEqual(length, 1)

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_torch_dataset_getitem_classification(self):
        mock_annotation = Mock(spec=Label)
        mock_annotation.type = AnnotationType.label
        mock_annotation.label = 0
        self.mock_media_item.annotations = [mock_annotation]

        dm_torch_dataset = DmTorchDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="classification",
            transform=self.transform,
        )
        item = dm_torch_dataset[0]

        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], int)

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_torch_dataset_getitem_detection(self):
        mock_annotation = Mock(spec=Bbox)
        mock_annotation.type = AnnotationType.bbox
        mock_annotation.as_dict.return_value = {"bbox": [0, 0, 2, 2]}
        self.mock_media_item.annotations = [mock_annotation]

        dm_torch_dataset = DmTorchDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="detection",
            transform=self.transform,
        )
        item = dm_torch_dataset[0]

        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], list)

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_torch_dataset_getitem_segmentation_polygon(self):
        mock_annotation = Mock(spec=Polygon)
        mock_annotation.type = AnnotationType.polygon
        mock_annotation.as_dict.return_value = {"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}
        self.mock_media_item.annotations = [mock_annotation]

        dm_torch_dataset = DmTorchDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="segmentation",
            transform=self.transform,
        )
        item = dm_torch_dataset[0]

        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], list)

    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_torch_dataset_getitem_segmentation_mask(self):
        mock_annotation = Mock(spec=Mask)
        mock_annotation.type = AnnotationType.mask
        mock_annotation.image = np.array([[1, 1], [0, 0]])
        self.mock_media_item.annotations = [mock_annotation]

        dm_torch_dataset = DmTorchDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="segmentation",
            transform=self.transform,
            target_transform=self.transform,
        )
        item = dm_torch_dataset[0]

        self.assertIsInstance(item[0], torch.Tensor)
        self.assertIsInstance(item[1], torch.Tensor)

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

            multi_framework_dataset = FrameworkConverter(
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

        multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="detection")
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", transform=transform
        )

        for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
            assert torch.equal(torch_item[0], dm_item[0])


class TfConverterTest(TestCase):
    def setUp(self):
        self.mock_dataset = Mock(spec=Dataset)
        self.mock_annotation = Mock(spec=Annotation)
        self.mock_media_item = Mock(spec=DatasetItem)
        self.mock_media_item.id = "0"
        self.mock_media_item.subset = "subset"
        self.mock_media_item.media = Image.from_numpy(np.array([[1, 2], [3, 4]]))
        self.mock_media_item.annotations = [self.mock_annotation]
        self.mock_dataset = Dataset.from_iterable([self.mock_media_item])
        self.output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

    # @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    # def test_tf_dataset_import(self):
    #     with self.assertRaises(ImportError):
    #         from datumaro.plugins.framework_converter import DmTfDataset

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_classification(self):
        mock_annotation = Mock(spec=Label)
        mock_annotation.type = AnnotationType.label
        mock_annotation.label = 0
        self.mock_media_item.annotations = [mock_annotation]
        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )

        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="classification",
            output_signature=output_signature,
        )
        tf_dataset = dm_tf_dataset.create_tf_dataset()
        self.assertIsInstance(tf_dataset, tf.data.Dataset)

        for item in tf_dataset:
            self.assertIsInstance(item[0], tf.Tensor)
            self.assertIsInstance(item[1], tf.Tensor)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_detection(self):
        mock_annotation = Mock(spec=Bbox)
        mock_annotation.type = AnnotationType.bbox
        mock_annotation.as_dict.return_value = {"bbox": [0, 0, 2, 2], "category_id": 0}
        self.mock_media_item.annotations = [mock_annotation]
        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            {
                "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="bbox"),
                "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="category_id"),
            },
        )

        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="detection",
            output_signature=output_signature,
        )
        tf_dataset = dm_tf_dataset.create_tf_dataset()
        self.assertIsInstance(tf_dataset, tf.data.Dataset)

        for item in tf_dataset:
            self.assertIsInstance(item[0], tf.Tensor)
            self.assertIsInstance(item[1], dict)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_segmentation_polygon(self):
        mock_annotation = Mock(spec=Polygon)
        mock_annotation.type = AnnotationType.polygon
        mock_annotation.as_dict.return_value = {
            "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
            "category_id": 1,
        }
        self.mock_media_item.annotations = [mock_annotation]
        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            {
                "polygon": tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32, name="polygon"),
                "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="category_id"),
            },
        )

        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="segmentation",
            output_signature=output_signature,
        )
        tf_dataset = dm_tf_dataset.create_tf_dataset()
        self.assertIsInstance(tf_dataset, tf.data.Dataset)

        for item in tf_dataset:
            self.assertIsInstance(item[0], tf.Tensor)
            self.assertIsInstance(item[1], dict)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_segmentation_mask(self):
        mock_annotation = Mock(spec=Mask)
        mock_annotation.type = AnnotationType.mask
        mock_annotation.image = np.array([[1, 1], [0, 0]])
        self.mock_media_item.annotations = [mock_annotation]
        output_signature = (
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        )

        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="segmentation",
            output_signature=output_signature,
        )
        tf_dataset = dm_tf_dataset.create_tf_dataset()
        self.assertIsInstance(tf_dataset, tf.data.Dataset)

        for item in tf_dataset:
            self.assertIsInstance(item[0], tf.Tensor)
            self.assertIsInstance(item[1], tf.Tensor)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_repeat(self):
        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="classification",
            output_signature=self.output_signature,
        )
        repeated_dataset = dm_tf_dataset.repeat(count=5)

        self.assertIsInstance(repeated_dataset, tf.data.Dataset)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_tf_dataset_batch(self):
        dm_tf_dataset = DmTfDataset(
            dataset=self.mock_dataset,
            subset="subset",
            task="classification",
            output_signature=self.output_signature,
        )
        batched_dataset = dm_tf_dataset.batch(batch_size=2)

        self.assertIsInstance(batched_dataset, tf.data.Dataset)

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

            multi_framework_dataset = FrameworkConverter(
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
