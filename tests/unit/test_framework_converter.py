# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
import tempfile
from typing import Any, Dict
from unittest import TestCase, skipIf

import numpy as np
import pytest

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Label,
    LabelCategories,
    Mask,
    Polygon,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, Table, TableRow
from datumaro.plugins.framework_converter import (
    TASK_ANN_TYPE,
    DmTfDataset,
    DmTorchDataset,
    FrameworkConverter,
    FrameworkConverterFactory,
    _MultiFrameworkDataset,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

try:
    import torch
    from torchtext.data.utils import get_tokenizer
    from torchtext.datasets import IMDB
    from torchtext.vocab import build_vocab_from_iterator
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
                    Label(1),
                    Label(2),
                    Bbox(0, 0, 2, 2, label=0, attributes={"occluded": True}),
                    Bbox(2, 2, 4, 4, label=1, attributes={"occluded": False}),
                    Polygon([0, 0, 0, 2, 2, 2, 2, 0], label=0, attributes={"occluded": True}),
                    Polygon([2, 2, 2, 4, 4, 4, 4, 4], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[0, 0, 0, 1, 1]] * 5),
                        label=1,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="1",
                subset="train",
                annotations=[
                    Label(1),
                    Label(3),
                    Bbox(1, 1, 2, 2, label=1, attributes={"occluded": True}),
                    Bbox(0, 0, 1, 1, label=2, attributes={"occluded": False}),
                    Bbox(2, 2, 4, 4, label=4, attributes={"occluded": True}),
                    Polygon([1, 1, 1, 2, 2, 2, 2, 2], label=1, attributes={"occluded": True}),
                    Mask(
                        image=np.array([[1, 1, 0, 0, 0]] * 5),
                        label=1,
                    ),
                    Mask(
                        image=np.array([[0, 0, 1, 1, 0]] * 5),
                        label=2,
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=3,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
            DatasetItem(
                id="2",
                subset="val",
                annotations=[
                    Label(2),
                    Label(3),
                    Bbox(0, 0, 1, 1, label=1, attributes={"occluded": False}),
                    Bbox(1, 1, 2, 2, label=2, attributes={"occluded": False}),
                    Bbox(2, 2, 4, 4, label=3, attributes={"occluded": True}),
                    Polygon([0, 0, 1, 0, 1, 1, 0, 1], label=2, attributes={"occluded": False}),
                    Mask(
                        image=np.array([[0, 1, 0, 0, 0]] * 5),
                        label=2,
                    ),
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=3,
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
                    Mask(
                        image=np.array([[0, 0, 0, 0, 1]] * 5),
                        label=2,
                    ),
                ],
                media=Image.from_numpy(data=np.ones((5, 5, 3))),
            ),
        ],
        categories={AnnotationType.label: label_cat},
    )


@pytest.fixture
def fxt_tabular_dataset():
    table = Table.from_list(
        [{"nswprice": 0.076108, "class": "DOWN"}, {"nswprice": 0.060376, "class": "UP"}]
    )
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="0",
                subset="train",
                media=TableRow(table=table, index=0),
                annotations=[Label(label=0), Caption("nswprice:0.076108")],
            ),
            DatasetItem(
                id="1",
                subset="train",
                media=TableRow(table=table, index=1),
                annotations=[Label(label=1), Caption("nswprice:0.060376")],
            ),
        ],
        categories={
            AnnotationType.label: LabelCategories.from_iterable(
                [("class:DOWN", "class"), ("class:UP", "class")]
            )
        },
        media_type=TableRow,
    )


@pytest.mark.new
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
    @pytest.mark.parametrize(
        "fxt_dataset_type,fxt_subset,fxt_task",
        [
            (
                "fxt_dataset",
                "train",
                "classification",
            ),
            (
                "fxt_dataset",
                "val",
                "multilabel_classification",
            ),
            (
                "fxt_dataset",
                "train",
                "detection",
            ),
            (
                "fxt_dataset",
                "val",
                "instance_segmentation",
            ),
            (
                "fxt_dataset",
                "train",
                "semantic_segmentation",
            ),
            ("fxt_tabular_dataset", "train", "tabular"),
        ],
    )
    def test_multi_framework_dataset(
        self, fxt_dataset_type: str, fxt_subset: str, fxt_task: str, request
    ):
        dataset = request.getfixturevalue(fxt_dataset_type)
        dm_multi_framework_dataset = _MultiFrameworkDataset(
            dataset=dataset, subset=fxt_subset, task=fxt_task
        )

        for idx in range(len(dm_multi_framework_dataset)):
            image, label = dm_multi_framework_dataset._gen_item(idx)
            assert isinstance(image, (np.ndarray, dict))
            if fxt_task == "classification":
                assert isinstance(label, int)
            elif fxt_task == "multilabel_classification":
                assert isinstance(label, list)
            if fxt_task in ["detection", "instance_segmentation"]:
                assert isinstance(label, list)
            if fxt_task == "semantic_segmentation":
                assert isinstance(label, np.ndarray)
            elif fxt_task == "tabular":
                assert isinstance(label, list)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    @pytest.mark.parametrize(
        "fxt_dataset_type,fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "fxt_dataset",
                "train",
                "classification",
                {},
            ),
            (
                "fxt_dataset",
                "val",
                "multilabel_classification",
                {},
            ),
            (
                "fxt_dataset",
                "train",
                "detection",
                {},
            ),
            (
                "fxt_dataset",
                "val",
                "instance_segmentation",
                {},
            ),
            (
                "fxt_dataset",
                "train",
                "semantic_segmentation",
                {},
            ),
            (
                "fxt_dataset",
                "val",
                "semantic_segmentation",
                {"transform": None, "target_transform": None},
            ),
            (
                "fxt_dataset",
                "train",
                "semantic_segmentation",
                {"transform": transforms.ToTensor()},
            ),
            (
                "fxt_dataset",
                "train",
                "tabular",
                {},
            ),
        ],
    )
    def test_can_convert_torch_framework(
        self,
        fxt_dataset_type: str,
        fxt_subset: str,
        fxt_task: str,
        fxt_convert_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
    ):
        dataset = request.getfixturevalue(fxt_dataset_type)
        multi_framework_dataset = FrameworkConverter(dataset, subset=fxt_subset, task=fxt_task)

        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch", **fxt_convert_kwargs
        )

        expected_dataset = dataset.get_subset(fxt_subset)

        for exp_item, dm_torch_item in zip(expected_dataset, dm_torch_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ][0]
            elif fxt_task == "multilabel_classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                masks = [
                    ann.as_class_mask()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = np.sum(masks, axis=0, dtype=np.uint8)
            elif fxt_task == "tabular":
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type in TASK_ANN_TYPE[fxt_task]
                ]
            if fxt_convert_kwargs.get("transform", None):
                actual = dm_torch_item[0].permute(1, 2, 0).mul(255.0).to(torch.uint8).numpy()
                assert np.array_equal(image, actual)
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

            assert len(torch_dataset) == len(dm_torch_dataset)
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

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
    def test_can_convert_torch_framework_tabular_label(self):
        class IMDBDataset(Dataset):
            def __init__(self, data_iter, vocab, transform=None):
                self._data = list(data_iter)
                self.vocab = vocab
                self.transform = transform
                self.tokenizer = get_tokenizer("basic_english")

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                label, text = self._data[idx]
                label = 1 if label == "pos" else 0
                tokens = self.tokenizer(text)
                token_ids = self.vocab(tokens)

                if self.transform:
                    token_ids = self.transform(token_ids)

                return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
                    label, dtype=torch.long
                )

        # data_path = osp.join(
        #     get_test_asset_path("tabular_dataset"), "women-clothing", "women_clothing_refined.csv"
        # )
        data_path = "/home/sooahlee/imdb_train.csv"
        train_iter = IMDB(split="train")
        tokenizer = get_tokenizer("basic_english")

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        torch_dataset = IMDBDataset(train_iter, vocab, transform=None)

        target = {"input": ["text"], "output": ["label"]}
        dm_dataset = Dataset.import_from(data_path, "tabular", target=target)
        dm_dataset = dm_dataset.transform("astype_annotations")

        multi_framework_dataset = FrameworkConverter(
            dm_dataset, subset="imdb_train", task="tabular"
        )
        dm_torch_dataset = multi_framework_dataset.to_framework(
            framework="torch",
        )

        for torch_item, dm_item in zip(torch_dataset, dm_torch_dataset):
            assert torch_item[1] == dm_item[0]["text"]

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_convert_kwargs",
        [
            (
                "train",
                "classification",
                {
                    "output_signature": {
                        "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                    }
                },
            ),
            (
                "val",
                "multilabel_classification",
                {
                    "output_signature": {
                        "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        "label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                    }
                },
            ),
            (
                "train",
                "detection",
                {
                    "output_signature": {
                        "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="points"),
                        "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="label"),
                    }
                },
            ),
            (
                "val",
                "instance_segmentation",
                {
                    "output_signature": {
                        "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        "polygon": tf.TensorSpec(
                            shape=(None, None), dtype=tf.float32, name="points"
                        ),
                        "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="label"),
                    }
                },
            ),
            (
                "train",
                "semantic_segmentation",
                {
                    "output_signature": {
                        "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                        "label": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                    }
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

        for exp_item, tf_item in zip(expected_dataset, dm_tf_dataset.create()):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            if fxt_task == "multilabel_classification":
                label = [
                    ann.label for ann in exp_item.annotations if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task in ["detection", "instance_segmentation"]:
                label = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
            elif fxt_task == "semantic_segmentation":
                masks = [
                    ann.as_class_mask()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = np.sum(masks, axis=0, dtype=np.uint8)

            assert np.array_equal(image, tf_item["image"])

            if fxt_task == "classification":
                assert label == tf_item["label"]

            if fxt_task == "multilabel_classification":
                assert np.array_equal(label, tf_item["label"])

            elif fxt_task == "detection":
                bboxes = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(bboxes, tf_item["bbox"].numpy())
                assert np.array_equal(labels, tf_item["category_id"].numpy())

            elif fxt_task == "instance_segmentation":
                polygons = [p["points"] for p in label]
                labels = [p["label"] for p in label]

                assert np.array_equal(polygons, tf_item["polygon"].numpy())
                assert np.array_equal(labels, tf_item["category_id"].numpy())

            elif fxt_task == "semantic_segmentation":
                assert np.array_equal(label, tf_item["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_output_signature",
        [
            (
                "train",
                "classification",
                {
                    "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            ),
            (
                "val",
                "detection",
                {
                    "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                    "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="points"),
                    "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="label"),
                },
            ),
        ],
    )
    def test_tf_get_rawitem(
        self, fxt_dataset: Dataset, fxt_subset: str, fxt_task: str, fxt_output_signature: dict
    ):
        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset=fxt_subset,
            task=fxt_task,
            output_signature=fxt_output_signature,
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for idx, exp_item in enumerate(expected_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            elif fxt_task == "detection":
                bboxes = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = []
                for key, spec in fxt_output_signature.items():
                    if key == "image":
                        continue
                    label += [tf.convert_to_tensor([bbox.get(spec.name, None) for bbox in bboxes])]

            tf_item = dm_tf_dataset._get_rawitem(idx)

            assert np.array_equal(image, tf_item[0])
            if fxt_task == "classification":
                assert label == tf_item[1]
            elif fxt_task == "detection":
                for label_types in range(len(label)):
                    assert np.array_equal(label[label_types], tf_item[label_types + 1])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    @pytest.mark.parametrize(
        "fxt_subset,fxt_task,fxt_output_signature",
        [
            (
                "train",
                "classification",
                {
                    "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                    "label": tf.TensorSpec(shape=(), dtype=tf.int32),
                },
            ),
            (
                "val",
                "detection",
                {
                    "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
                    "bbox": tf.TensorSpec(shape=(None, 4), dtype=tf.float32, name="points"),
                    "category_id": tf.TensorSpec(shape=(None,), dtype=tf.int32, name="label"),
                },
            ),
        ],
    )
    def test_tf_process_item(
        self, fxt_dataset: Dataset, fxt_subset: str, fxt_task: str, fxt_output_signature: dict
    ):
        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset=fxt_subset,
            task=fxt_task,
            output_signature=fxt_output_signature,
        )

        expected_dataset = fxt_dataset.get_subset(fxt_subset)

        for idx, exp_item in enumerate(expected_dataset):
            image = exp_item.media.data
            if fxt_task == "classification":
                label = exp_item.annotations[0].label
            elif fxt_task == "detection":
                bboxes = [
                    ann.as_dict()
                    for ann in exp_item.annotations
                    if ann.type == TASK_ANN_TYPE[fxt_task]
                ]
                label = []
                for key, spec in fxt_output_signature.items():
                    if key == "image":
                        continue
                    label += [tf.convert_to_tensor([bbox.get(spec.name, None) for bbox in bboxes])]

            tf_item = dm_tf_dataset._process_item(idx)

            assert np.array_equal(image, tf_item["image"])
            if fxt_task == "classification":
                assert label == tf_item["label"]
            elif fxt_task == "detection":
                assert np.array_equal(label[0], tf_item["bbox"])
                assert np.array_equal(label[1], tf_item["category_id"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_tf_dataset_repeat(self, fxt_dataset: Dataset):
        output_signature = {
            "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset="train",
            task="classification",
            output_signature=output_signature,
        )
        original_dataset = dm_tf_dataset.create()
        repeated_dataset = dm_tf_dataset.repeat(count=5)

        n_dataset = len(list(original_dataset))

        for idx, item in enumerate(repeated_dataset):
            assert np.array_equal(item["image"], list(original_dataset)[idx % n_dataset]["image"])
            assert np.array_equal(item["label"], list(original_dataset)[idx % n_dataset]["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_tf_dataset_batch(self, fxt_dataset: Dataset):
        output_signature = {
            "image": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            "label": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        }

        dm_tf_dataset = DmTfDataset(
            dataset=fxt_dataset,
            subset="val",
            task="semantic_segmentation",
            output_signature=output_signature,
        )
        original_dataset = dm_tf_dataset.create()
        batched_dataset = dm_tf_dataset.batch(batch_size=2)

        for idx, item in enumerate(batched_dataset):
            assert np.array_equal(item["image"][0], list(original_dataset)[idx]["image"])
            assert np.array_equal(item["image"][1], list(original_dataset)[idx + 1]["image"])
            assert np.array_equal(item["label"][0], list(original_dataset)[idx]["label"])
            assert np.array_equal(item["label"][1], list(original_dataset)[idx + 1]["label"])

    @pytest.mark.skipif(not TF_AVAILABLE, reason="Tensorflow is not installed")
    def test_can_convert_tf_framework_classification(self):
        output_signature = {
            "image": tf.TensorSpec(shape=(28, 28), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.uint8),
        }

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
                assert tf.reduce_all(tf_item[0] == dm_item["image"])
                assert tf.reduce_all(tf_item[1] == dm_item["label"])

    @pytest.mark.skipif(TORCH_AVAILABLE, reason="PyTorch is installed")
    def test_torch_dataset_import(self):
        with pytest.raises(ImportError):
            from datumaro.plugins.framework_converter import DmTorchDataset

    @pytest.mark.skipif(TF_AVAILABLE, reason="Tensorflow is installed")
    def test_tf_dataset_import(self):
        with pytest.raises(ImportError):
            from datumaro.plugins.framework_converter import DmTfDataset
