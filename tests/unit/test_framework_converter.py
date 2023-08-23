# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from typing import Any, Dict
from unittest import TestCase, skipIf
from unittest.mock import Mock, patch

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    Polygon,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.framework_converter import (
    TASK_ANN_TYPE,
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


@pytest.fixture
def fxt_dataset():
    label_cat = LabelCategories.from_iterable([f"label_{i}" for i in range(4)])
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="train",
                annotations=[
                    Label(0),
                    Bbox(0, 0, 2, 2, label=0, attributes={"occluded": True}),
                    Bbox(2, 2, 4, 4, label=1, attributes={"occluded": False}),
                    Polygon([0, 0, 0, 2, 2, 2, 2, 0], label=0, attributes={"occluded": True}),
                    Polygon([2, 2, 2, 4, 4, 4, 4, 4], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        label=0,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="train",
                annotations=[
                    Label(1),
                    Bbox(1, 1, 2, 2, label=1, attributes={"occluded": True}),
                    Polygon([1, 1, 1, 2, 2, 2, 2, 2], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        label=1,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="val",
                annotations=[
                    Label(2),
                    Bbox(0, 0, 1, 1, label=2, attributes={"occluded": False}),
                    Polygon([0, 0, 1, 0, 1, 1, 0, 1], label=2, attributes={"occluded": False}),
                    Mask(
                        image=np.array([[0, 1, 0, 0, 0]] * 5),
                        label=2,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="3",
                subset="val",
                annotations=[
                    Label(3),
                    Bbox(1, 1, 4, 4, label=3, attributes={"occluded": True}),
                    Polygon([1, 1, 1, 4, 4, 4, 4, 1], label=3, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[1, 1, 1, 1, 0]] * 5),
                        label=3,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat},
    )


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
class FrameworkConverterFactoryTest(TestCase):
    @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_create_converter_torch(self):
        converter = FrameworkConverterFactory.create_converter("torch")
        self.assertEqual(converter, DmTorchDataset)

    @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_create_converter_tf(self):
        converter = FrameworkConverterFactory.create_converter("tf")
        self.assertEqual(converter, DmTfDataset)

    def test_create_converter_invalid(self):
        with self.assertRaises(ValueError):
            FrameworkConverterFactory.create_converter("invalid_framework")

    @skipIf(TORCH_AVAILABLE, reason="PyTorch is installed")
    def test_create_converter_torch_importerror(self):
        with self.assertRaises(ImportError):
            _ = FrameworkConverterFactory.create_converter("torch")

    @skipIf(TF_AVAILABLE, reason="Tensorflow is installed")
    def test_create_converter_tf_importerror(self):
        with self.assertRaises(ImportError):
            _ = FrameworkConverterFactory.create_converter("tf")


@mark_requirement(Requirements.DATUM_GENERAL_REQ)
class MultiframeworkConverterTest:
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "train",
                "classification",
                {},
            ),
            (
                "val",
                "classification",
                {},
            ),
            (
                "train",
                "detection",
                {},
            ),
            (
                "val",
                "instance_segmentation",
                {},
            ),
            (
                "train",
                "semantic_segmentation",
                {},
            ),
            (
                "val",
                "semantic_segmentation",
                {"transform": None, "target_transform": None},
            ),
            (
                "train",
                "classification",
                {"transform": transforms.ToTensor()},
            ),
            (
                "val",
                "semantic_segmentation",
                {"transform": transforms.ToTensor(), "target_transform": transforms.ToTensor()},
            ),
        ],
    )
    def test_can_convert_torch_framework(
        self,
        fxt_dataset: Dataset,
        fxt_subset: str,
        fxt_task: str,
        fxt_convert_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        multi_framework_dataset = FrameworkConverter(fxt_dataset, subset=fxt_subset, task=fxt_task)

        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", **fxt_convert_kwargs
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for exp_item, dm_torch_item in zip(expected_dataset, dm_torch_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ][0]
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                label = [
                    ann.image for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ][0]

            if fxt_convert_kwargs.get("transform", None):
                assert np.array_equal(image, dm_torch_item[0].reshape(5, 5, 3).numpy())
            else:
                assert np.array_equal(image, dm_torch_item[0])

            if fxt_convert_kwargs.get("target_transform", None):
                assert np.array_equal(label, dm_torch_item[1].squeeze(0).numpy())
            else:
                assert np.array_equal(label, dm_torch_item[1])

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_classification(self):
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
                assert torch_item[1] == dm_item[1]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_detection(self):
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
            for torch_ann, dm_ann in zip(torch_item[1], dm_item[1]):
                assert torch_ann["id"] == dm_ann["id"]
                assert torch_ann["category_id"] == dm_ann["label"] + 1
                # torch: (x, y, w, h), while dm: (x1, y1, x2, y2)
                x1, y1, x2, y2 = dm_ann["points"]
                assert torch_ann["bbox"] == [x1, y1, x2 - x1, y2 - y1]
                assert torch_ann["iscrowd"] == dm_ann["attributes"]["is_crowd"]

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "train",
                "classification",
                {
                    "output_signature": (
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(), dtype=tf.int32),
                    )
                },
            ),
            (
                "val",
                "detection",
                {
                    "output_signature": (
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        {
                            "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="points"),
                            "category_id": tf.TensorSpec(
                                shape=(None,), dtype=tf.int32, name="label"
                            ),
                        },
                    )
                },
            ),
            (
                "train",
                "instance_segmentation",
                {
                    "output_signature": (
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        {
                            "polygon": tf.TensorSpec(
                                shape=(None, None), dtype=tf.float32, name="points"
                            ),
                            "category_id": tf.TensorSpec(
                                shape=(None,), dtype=tf.int32, name="label"
                            ),
                        },
                    )
                },
            ),
            (
                "val",
                "semantic_segmentation",
                {
                    "output_signature": (
                        tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    )
                },
            ),
        ],
    )
    def test_can_convert_tf_framework(
        self,
        fxt_dataset: Dataset,
        fxt_subset: str,
        fxt_task: str,
        fxt_convert_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        multi_framework_dataset = FrameworkConverter(fxt_dataset, subset=fxt_subset, task=fxt_task)

        dm_tf_dataset = multi_framework_dataset.to_framework(framework="tf", **fxt_convert_kwargs)

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for exp_item, tf_item in zip(expected_dataset, dm_tf_dataset.create_tf_dataset()):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                label = [
                    ann.image for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ][0]

            assert np.array_equal(image, tf_item[0])

            if fxt_task == "classification":
                assert label == tf_item[1]

            elif fxt_task == "detection":
                bboxes = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(bboxes, tf_item[1]["bbox"].numpy())
                assert np.array_equal(labels, tf_item[1]["category_id"].numpy())

            elif fxt_task == "instance_segmentation":
                polygons = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(polygons, tf_item[1]["polygon"].numpy())
                assert np.array_equal(labels, tf_item[1]["category_id"].numpy())

            elif fxt_task == "semantic_segmentation":
                assert np.array_equal(label, tf_item[1])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_can_convert_tf_framework_classification(self):
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

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="PyTorch is installed")
    def test_torch_dataset_import(self):
        with pytest.raises(ImportError):
            from datumaro.plugins.framework_converter import DmTorchDataset

    @pytest.mark.skipif(TF_AVAILABLE, reason="Tensorflow is installed")
    def test_tf_dataset_import(self):
        with pytest.raises(ImportError):
            from datumaro.plugins.framework_converter import DmTfDataset


# @skipIf(not TORCH_AVAILABLE, reason="PyTorch is not installed")
# class TorchConverterTest(TestCase):
#     def setUp(self):
#         self.mock_dataset = Mock(spec=Dataset)
#         self.mock_annotation = Mock(spec=Annotation)
#         self.mock_media_item = Mock(spec=DatasetItem)
#         self.mock_media_item.id = "0"
#         self.mock_media_item.subset = "subset"
#         self.mock_media_item.media = Image.from_numpy(np.array([[0, 0], [1, 1]]))
#         self.mock_media_item.annotations = [self.mock_annotation]
#         self.mock_dataset = Dataset.from_iterable([self.mock_media_item])
#         self.transform = transforms.ToTensor()

#     # @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     # def test_torch_dataset_import(self):
#     #     with patch.dict('sys.modules', {'torch': None}):
#     #         with self.assertRaises(ImportError):
#     #             DmTorchDataset(
#     #                 dataset=self.mock_dataset,
#     #                 subset="subset",
#     #                 task="classification",
#     #             )

#     def test_torch_dataset_init(self):
#         transform_fn = lambda x: x + 1
#         target_transform_fn = lambda y: y * 2
#         dm_dataset = DmTorchDataset(
#             self.mock_dataset,
#             subset="subset",
#             task="classification",
#             transform=transform_fn,
#             target_transform=target_transform_fn,
#         )
#         compare_datasets(self, dm_dataset.dataset, self.mock_dataset.get_subset("subset"))
#         self.assertEqual(dm_dataset.subset, "subset")
#         self.assertEqual(dm_dataset.task, "classification")
#         self.assertEqual(dm_dataset.transform, transform_fn)
#         self.assertEqual(dm_dataset.target_transform, target_transform_fn)

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_torch_dataset_len(self):
#         dm_torch_dataset = DmTorchDataset(
#             dataset=self.mock_dataset,
#             subset="subset",
#             task="classification",
#         )
#         length = len(dm_torch_dataset)

#         self.assertEqual(length, 1)

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_torch_dataset_getitem_classification(self):
#         mock_annotation = Mock(spec=Label)
#         mock_annotation.type = AnnotationType.label
#         mock_annotation.label = 0
#         self.mock_media_item.annotations = [mock_annotation]

#         dm_torch_dataset = DmTorchDataset(
#             dataset=self.mock_dataset,
#             subset="subset",
#             task="classification",
#             transform=self.transform,
#         )
#         item = dm_torch_dataset[0]

#         assert torch.equal(item[0], torch.tensor([[[0, 0], [1, 1]]]))
#         self.assertEqual(item[1], 0)

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_torch_dataset_getitem_detection(self):
#         mock_annotation = Mock(spec=Bbox)
#         mock_annotation.type = AnnotationType.bbox
#         mock_annotation.as_dict.return_value = {"bbox": [0, 0, 2, 2]}
#         self.mock_media_item.annotations = [mock_annotation]

#         dm_torch_dataset = DmTorchDataset(
#             dataset=self.mock_dataset,
#             subset="subset",
#             task="detection",
#             transform=self.transform,
#         )
#         item = dm_torch_dataset[0]

#         assert torch.equal(item[0], torch.tensor([[[0, 0], [1, 1]]]))
#         self.assertEqual(item[1], [{"bbox": [0, 0, 2, 2]}])

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_torch_dataset_getitem_segmentation_polygon(self):
#         mock_annotation = Mock(spec=Polygon)
#         mock_annotation.type = AnnotationType.polygon
#         mock_annotation.as_dict.return_value = {"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}
#         self.mock_media_item.annotations = [mock_annotation]

#         dm_torch_dataset = DmTorchDataset(
#             dataset=self.mock_dataset,
#             subset="subset",
#             task="segmentation",
#             transform=self.transform,
#         )
#         item = dm_torch_dataset[0]

#         assert torch.equal(item[0], torch.tensor([[[0, 0], [1, 1]]]))
#         self.assertEqual(item[1], [{"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}])

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_torch_dataset_getitem_segmentation_mask(self):
#         mock_annotation = Mock(spec=Mask)
#         mock_annotation.type = AnnotationType.mask
#         mock_annotation.image = np.array([[1, 1], [0, 0]])
#         self.mock_media_item.annotations = [mock_annotation]

#         dm_torch_dataset = DmTorchDataset(
#             dataset=self.mock_dataset,
#             subset="subset",
#             task="segmentation",
#             transform=self.transform,
#             target_transform=self.transform,
#         )
#         item = dm_torch_dataset[0]

#         assert torch.equal(item[0], torch.tensor([[[0, 0], [1, 1]]]))
#         assert torch.equal(item[1], torch.tensor([[[1, 1], [0, 0]]]))

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_can_convert_torch_classification_data(self):
#         transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Resize((64, 64)),
#             ]
#         )

#         with tempfile.TemporaryDirectory() as tmp_dir:
#             torch_dataset = datasets.MNIST(
#                 root=tmp_dir,
#                 train=True,
#                 download=True,
#                 transform=transform,
#             )

#             dm_dataset = Dataset.import_from(path=osp.join(tmp_dir, "MNIST"), format="mnist")

#             multi_framework_dataset = FrameworkConverter(
#                 dm_dataset, subset="train", task="classification"
#             )
#             dm_torch_dataset = multi_framework_dataset.to_framework(
#                 framework="torch", transform=transform
#             )

#             for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
#                 assert torch.equal(torch_item[0], dm_item[0])
#                 self.assertEqual(torch_item[1], dm_item[1])

#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     def test_can_convert_torch_detection_data(self):
#         DUMMY_DATASET_DIR = get_test_asset_path("coco_dataset")
#         format = "coco_instances"

#         data_path = osp.join(DUMMY_DATASET_DIR, format)

#         transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Resize((256, 256)),
#             ]
#         )

#         torch_dataset = datasets.CocoDetection(
#             root=osp.join(data_path, "images/train/"),
#             annFile=osp.join(data_path, "annotations/instances_train.json"),
#             transform=transform,
#         )

#         dm_dataset = Dataset.import_from(data_path, format)

#         multi_framework_dataset = FrameworkConverter(dm_dataset, subset="train", task="detection")
#         dm_torch_dataset = multi_framework_dataset.to_framework(
#             framework="torch", transform=transform
#         )

#         for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
#             assert torch.equal(torch_item[0], dm_item[0])


# @skipIf(not TF_AVAILABLE, reason="Tensorflow is not installed")
# class TfConverterTest(TestCase):
#     def setUp(self):
#         self.mock_dataset = Mock(spec=Dataset)
#         self.mock_annotation = Mock(spec=Annotation)
#         self.mock_media_item = Mock(spec=DatasetItem)
#         self.mock_media_item.id = "0"
#         self.mock_media_item.subset = "subset"
#         self.mock_media_item.media = Image.from_numpy(np.array([[0, 0], [1, 1]]))
#         self.mock_media_item.annotations = [self.mock_annotation]
#         self.mock_dataset = Dataset.from_iterable([self.mock_media_item])
#         self.output_signature = (
#             tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#             tf.TensorSpec(shape=(None,), dtype=tf.int32),
#         )


# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_generator_wrapper(self):
#     mock_annotation = Mock()
#     mock_annotation.as_dict.return_value = {"key1": 10, "key2": 3.14}
#     self.mock_media_item.annotations = [mock_annotation]
#     output_signature = (
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         {
#             "key1": tf.TensorSpec(shape=(), dtype=tf.int32, name="key1"),
#             "key2": tf.TensorSpec(shape=(), dtype=tf.float32, name="key2"),
#         },
#     )

#     converter = DmTfDataset(
#         dataset=self.mock_dataset,
#         task="detection",
#         subset="subset",
#         output_signature=output_signature,
#     )
#     generator = converter.generator_wrapper()

#     image, label = next(generator)

#     assert np.array_equal(image, np.array([[0, 0], [1, 1]]))
#     self.assertIsInstance(label, dict)
#     self.assertEqual(label["key1"], [10])
#     self.assertEqual(label["key2"], [3.14])

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_classification(self):
#     mock_annotation = Mock(spec=Label)
#     mock_annotation.type = AnnotationType.label
#     mock_annotation.label = 0
#     self.mock_media_item.annotations = [mock_annotation]
#     output_signature = (
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         tf.TensorSpec(shape=(), dtype=tf.int32),
#     )

#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="classification",
#         output_signature=output_signature,
#     )
#     tf_dataset = dm_tf_dataset.create_tf_dataset()
#     self.assertIsInstance(tf_dataset, tf.data.Dataset)

#     for item in tf_dataset:
#         self.assertIsInstance(item[0], tf.Tensor)
#         assert np.array_equal(item[0], np.array([[0, 0], [1, 1]]))
#         self.assertIsInstance(item[1], tf.Tensor)
#         assert np.array_equal(item[1], 0)

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_detection(self):
#     mock_annotation = Mock(spec=Bbox)
#     mock_annotation.type = AnnotationType.bbox
#     mock_annotation.as_dict.return_value = {"bbox": [0, 0, 2, 2], "category_id": 0}
#     self.mock_media_item.annotations = [mock_annotation]
#     output_signature = (
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         {
#             "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="bbox"),
#             "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="category_id"),
#         },
#     )

#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="detection",
#         output_signature=output_signature,
#     )
#     tf_dataset = dm_tf_dataset.create_tf_dataset()
#     self.assertIsInstance(tf_dataset, tf.data.Dataset)

#     for item in tf_dataset:
#         self.assertIsInstance(item[0], tf.Tensor)
#         assert np.array_equal(item[0], np.array([[0, 0], [1, 1]]))
#         self.assertIsInstance(item[1], dict)
#         assert np.array_equal(item[1]["bbox"], [[0, 0, 2, 2]])
#         assert np.array_equal(item[1]["category_id"], [0])

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_segmentation_polygon(self):
#     mock_annotation = Mock(spec=Polygon)
#     mock_annotation.type = AnnotationType.polygon
#     mock_annotation.as_dict.return_value = {
#         "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
#         "category_id": 1,
#     }
#     self.mock_media_item.annotations = [mock_annotation]
#     output_signature = (
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         {
#             "polygon": tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32, name="polygon"),
#             "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="category_id"),
#         },
#     )

#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="segmentation",
#         output_signature=output_signature,
#     )
#     tf_dataset = dm_tf_dataset.create_tf_dataset()
#     self.assertIsInstance(tf_dataset, tf.data.Dataset)

#     for item in tf_dataset:
#         self.assertIsInstance(item[0], tf.Tensor)
#         assert np.array_equal(item[0], np.array([[0, 0], [1, 1]]))
#         self.assertIsInstance(item[1], dict)
#         assert np.array_equal(item[1]["polygon"], [[[0, 0], [2, 0], [2, 2], [0, 2]]])
#         assert np.array_equal(item[1]["category_id"], [1])

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_segmentation_mask(self):
#     mock_annotation = Mock(spec=Mask)
#     mock_annotation.type = AnnotationType.mask
#     mock_annotation.image = np.array([[1, 1], [0, 0]])
#     self.mock_media_item.annotations = [mock_annotation]
#     output_signature = (
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#         tf.TensorSpec(shape=(None, None), dtype=tf.int32),
#     )

#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="segmentation",
#         output_signature=output_signature,
#     )
#     tf_dataset = dm_tf_dataset.create_tf_dataset()
#     self.assertIsInstance(tf_dataset, tf.data.Dataset)

#     for item in tf_dataset:
#         self.assertIsInstance(item[0], tf.Tensor)
#         assert np.array_equal(item[0], np.array([[0, 0], [1, 1]]))
#         self.assertIsInstance(item[1], tf.Tensor)
#         assert np.array_equal(item[1], np.array([[1, 1], [0, 0]]))

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_repeat(self):
#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="classification",
#         output_signature=self.output_signature,
#     )
#     repeated_dataset = dm_tf_dataset.repeat(count=5)

#     self.assertIsInstance(repeated_dataset, tf.data.Dataset)

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_tf_dataset_batch(self):
#     dm_tf_dataset = DmTfDataset(
#         dataset=self.mock_dataset,
#         subset="subset",
#         task="classification",
#         output_signature=self.output_signature,
#     )
#     batched_dataset = dm_tf_dataset.batch(batch_size=2)

#     self.assertIsInstance(batched_dataset, tf.data.Dataset)

# @mark_requirement(Requirements.DATUM_GENERAL_REQ)
# def test_can_convert_tf_classification_data(self):
#     output_signature = (
#         tf.TensorSpec(shape=(28, 28), dtype=tf.uint8),  # Modify shape and dtype accordingly
#         tf.TensorSpec(shape=(), dtype=tf.uint8),
#     )

#     with tempfile.TemporaryDirectory() as tmp_dir:
#         (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#         tf_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#         keras_data_dir = osp.expanduser("~/.keras/datasets")
#         dm_dataset = Dataset.import_from(
#             path=osp.join(keras_data_dir, "fashion-mnist"), format="mnist"
#         )

#         multi_framework_dataset = FrameworkConverter(
#             dm_dataset, subset="test", task="classification"
#         )
#         dm_tf_dataset = multi_framework_dataset.to_framework(
#             framework="tf", output_signature=output_signature
#         )

#         epoch, batch_size = 1, 16
#         for tf_item, dm_item in zip(
#             tf_dataset.repeat(epoch).batch(batch_size),
#             dm_tf_dataset.repeat(epoch).batch(batch_size),
#         ):
#             assert tf.reduce_all(tf_item[0] == dm_item[0])
#             assert tf.reduce_all(tf_item[1] == dm_item[1])
