# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, Label, Polygon
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.plugins.data_formats.roboflow.importer import (
    RoboflowCocoImporter,
    RoboflowVocImporter,
    RoboflowYoloImporter,
)

from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_COCO_DIR = get_test_asset_path("roboflow_dataset", "coco")
DUMMY_DATASET_VOC_DIR = get_test_asset_path("roboflow_dataset", "voc")
DUMMY_DATASET_YOLO_DIR = get_test_asset_path("roboflow_dataset", "yolo")
DUMMY_DATASET_YOLO_OBB_DIR = get_test_asset_path("roboflow_dataset", "yolo_obb")
DUMMY_DATASET_YOLO_OBB_DIR = get_test_asset_path("roboflow_dataset", "createml")
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


IDS = ["COCO", "VOC", "YOLO"]


class RoboflowImporterTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "importer"],
        [
            (DUMMY_DATASET_COCO_DIR, RoboflowCocoImporter),
            (DUMMY_DATASET_VOC_DIR, RoboflowVocImporter),
            (DUMMY_DATASET_YOLO_DIR, RoboflowYoloImporter),
        ],
        ids=IDS,
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        return super().test_can_detect(fxt_dataset_dir, importer)

    @pytest.mark.parametrize(
        ["fxt_dataset_dir", "fxt_expected_dataset", "fxt_import_kwargs", "importer"],
        [
            (DUMMY_DATASET_COCO_DIR, "fxt_coco_dataset", {}, RoboflowCocoImporter),
            (DUMMY_DATASET_VOC_DIR, "fxt_voc_dataset", {}, RoboflowVocImporter),
            (DUMMY_DATASET_YOLO_DIR, "fxt_yolo_dataset", {}, RoboflowYoloImporter),
        ],
        indirect=["fxt_expected_dataset"],
        ids=IDS,
    )
    def test_can_import(
        self,
        fxt_dataset_dir: str,
        fxt_expected_dataset: Dataset,
        fxt_import_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        importer: Importer,
    ):
        return super().test_can_import(
            fxt_dataset_dir,
            fxt_expected_dataset,
            fxt_import_kwargs,
            request,
            importer,
        )
