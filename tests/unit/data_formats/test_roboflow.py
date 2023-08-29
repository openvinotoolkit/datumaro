# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, Label, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.roboflow.importer import (
    RoboflowCocoImporter,
    RoboflowCreateMlImporter,
    RoboflowMulticlassImporter,
    RoboflowTfrecordImporter,
    RoboflowVocImporter,
    RoboflowYoloImporter,
    RoboflowYoloObbImporter,
)

from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path

try:
    import tensorflow as tf
except ImportError:
    TF_AVAILABLE = False
else:
    TF_AVAILABLE = True


DUMMY_DATASET_COCO_DIR = get_test_asset_path("roboflow_dataset", "coco")
DUMMY_DATASET_VOC_DIR = get_test_asset_path("roboflow_dataset", "voc")
DUMMY_DATASET_YOLO_DIR = get_test_asset_path("roboflow_dataset", "yolo")
DUMMY_DATASET_YOLO_OBB_DIR = get_test_asset_path("roboflow_dataset", "yolo_obb")
DUMMY_DATASET_CREATEML_DIR = get_test_asset_path("roboflow_dataset", "createml")
DUMMY_DATASET_MULTICLASS_DIR = get_test_asset_path("roboflow_dataset", "multiclass")
DUMMY_DATASET_TFRECORD_DIR = get_test_asset_path("roboflow_dataset", "tfrecord")


@pytest.fixture
def fxt_coco_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="train_001",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(2, 1, 3, 1, label=0, group=0, id=0, attributes={"is_crowd": False})
                ],
                attributes={"id": 1},
            ),
            DatasetItem(
                id="train_002",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(0, 0, 2, 4, label=1, group=1, id=1, attributes={"is_crowd": False})
                ],
                attributes={"id": 2},
            ),
            DatasetItem(
                id="val_001",
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(0, 0, 1, 2, label=0, group=1, id=1, attributes={"is_crowd": False}),
                    Bbox(0, 0, 9, 1, label=1, group=2, id=2, attributes={"is_crowd": True}),
                ],
                attributes={"id": 1},
            ),
        ],
        categories=["label_0", "label_1"],
    )


@pytest.fixture
def fxt_voc_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="train_001",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(
                        2,
                        1,
                        3,
                        1,
                        label=0,
                        group=0,
                        id=0,
                        attributes={"difficult": False, "truncated": False, "occluded": False},
                    )
                ],
            ),
            DatasetItem(
                id="train_002",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        2,
                        4,
                        label=1,
                        group=0,
                        id=0,
                        attributes={"difficult": False, "truncated": False, "occluded": False},
                    )
                ],
            ),
            DatasetItem(
                id="val_001",
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Bbox(
                        0,
                        0,
                        1,
                        2,
                        label=0,
                        group=0,
                        id=0,
                        attributes={"difficult": False, "truncated": False, "occluded": False},
                    ),
                    Bbox(
                        0,
                        0,
                        9,
                        1,
                        label=1,
                        group=1,
                        id=1,
                        attributes={"difficult": False, "truncated": False, "occluded": False},
                    ),
                ],
            ),
        ],
        categories=["label_0", "label_1"],
    )


@pytest.fixture
def fxt_yolo_dataset(fxt_voc_dataset):
    yolo_dataset = deepcopy(fxt_voc_dataset)
    yolo_dataset.transform("remove_attributes")
    return yolo_dataset


@pytest.fixture
def fxt_yolo_obb_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="train_001",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Polygon(
                        points=[0, 0, 0, 2, 2, 2, 2, 0],
                        label=0,
                        group=0,
                        id=0,
                    )
                ],
            ),
            DatasetItem(
                id="train_002",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Polygon(
                        points=[1, 1, 1, 5, 5, 5, 5, 1],
                        label=1,
                        group=0,
                        id=0,
                    )
                ],
            ),
            DatasetItem(
                id="val_001",
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Polygon(
                        points=[0, 0, 0, 1, 1, 1, 1, 0],
                        label=0,
                        group=0,
                        id=0,
                    ),
                    Polygon(
                        points=[1, 2, 1, 5, 10, 5, 10, 2],
                        label=1,
                        group=1,
                        id=1,
                    ),
                ],
            ),
        ],
        categories=["label_0", "label_1"],
    )


@pytest.fixture
def fxt_createml_dataset(fxt_yolo_dataset):
    createml_dataset = deepcopy(fxt_yolo_dataset)
    return createml_dataset


@pytest.fixture
def fxt_multiclass_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="train_001",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Label(label=0, group=0, id=0),
                ],
            ),
            DatasetItem(
                id="train_002",
                subset="train",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Label(label=1, group=0, id=0),
                ],
            ),
            DatasetItem(
                id="val_001",
                subset="val",
                media=Image.from_numpy(data=np.ones((5, 10, 3))),
                annotations=[
                    Label(label=0, group=0, id=0),
                    Label(label=1, group=1, id=1),
                ],
            ),
        ],
        categories=["label_0", "label_1"],
    )


IDS = ["COCO", "VOC", "YOLO", "YOLO_OBB", "CREATE_ML", "MULTICLASS"]


class RoboflowImporterTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "importer"],
        [
            (DUMMY_DATASET_COCO_DIR, RoboflowCocoImporter),
            (DUMMY_DATASET_VOC_DIR, RoboflowVocImporter),
            (DUMMY_DATASET_YOLO_DIR, RoboflowYoloImporter),
            (DUMMY_DATASET_YOLO_OBB_DIR, RoboflowYoloObbImporter),
            (DUMMY_DATASET_CREATEML_DIR, RoboflowCreateMlImporter),
            (DUMMY_DATASET_MULTICLASS_DIR, RoboflowMulticlassImporter),
        ],
        ids=IDS,
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        return super().test_can_detect(fxt_dataset_dir, importer)

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "importer"],
        [
            (DUMMY_DATASET_COCO_DIR, "fxt_coco_dataset", RoboflowCocoImporter),
            (DUMMY_DATASET_VOC_DIR, "fxt_voc_dataset", RoboflowVocImporter),
            (DUMMY_DATASET_YOLO_DIR, "fxt_yolo_dataset", RoboflowYoloImporter),
            (DUMMY_DATASET_YOLO_OBB_DIR, "fxt_yolo_obb_dataset", RoboflowYoloObbImporter),
            (DUMMY_DATASET_CREATEML_DIR, "fxt_createml_dataset", RoboflowCreateMlImporter),
            (DUMMY_DATASET_MULTICLASS_DIR, "fxt_multiclass_dataset", RoboflowMulticlassImporter),
        ],
        indirect=["fxt_expected_dataset"],
        ids=IDS,
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir=fxt_dataset_dir,
            fxt_expected_dataset=fxt_expected_dataset,
            fxt_import_kwargs={},
            request=request,
            importer=importer,
        )

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "importer"],
        [
            (DUMMY_DATASET_TFRECORD_DIR, RoboflowTfrecordImporter),
        ],
    )
    def test_can_detect_roboflow_tfrecord(self, fxt_dataset_dir: str, importer: Importer):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert importer.NAME in detected_formats

    @pytest.mark.skipif(TF_AVAILABLE, reason="Tensorflow is installed")
    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "importer"],
        [
            (DUMMY_DATASET_TFRECORD_DIR, "fxt_coco_dataset", RoboflowTfrecordImporter),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_import_roboflow_tfrecord(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir=fxt_dataset_dir,
            fxt_expected_dataset=fxt_expected_dataset,
            fxt_import_kwargs={},
            request=request,
            importer=importer,
        )
