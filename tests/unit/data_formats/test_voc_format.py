# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from typing import Any, Dict

import numpy as np
from lxml import etree as ElementTree  # nosec
import pytest

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image

from datumaro.plugins.data_formats.voc.exporter import (
    VocActionExporter,
    VocClassificationExporter,
    VocDetectionExporter,
    VocLayoutExporter,
    VocSegmentationExporter,
)
from datumaro.plugins.data_formats.voc.format import VocTask
from datumaro.plugins.data_formats.voc.importer import (
    VocActionImporter,
    VocClassificationImporter,
    VocDetectionImporter,
    VocLayoutImporter,
    VocSegmentationImporter,
)

from .base import TestDataFormatBase

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_DIR = get_test_asset_path("voc_dataset", "voc_dataset1")
DUMMY_DATASET2_DIR = get_test_asset_path("voc_dataset", "voc_dataset2")
DUMMY_DATASET3_DIR = get_test_asset_path("voc_dataset", "voc_dataset3")

@pytest.fixture
def fxt_classification_dataset():
    return Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Label(label=l) for l in range(len(VOC.VocLabel)) if l % 2 == 1
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.classification),
        )

@pytest.fixture
def fxt_detection_dataset():
    return Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            1.0,
                            2.0,
                            2.0,
                            2.0,
                            label=8,
                            id=0,
                            group=0,
                            attributes={
                                "difficult": False,
                                "truncated": True,
                                "occluded": False,
                                "pose": "Unspecified",
                            },
                        ),
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=15,
                            id=1,
                            group=1,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.detection),
        )

@pytest.fixture
def fxt_segmentation_dataset():
    return Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[Mask(image=np.ones([10, 20]), label=2, group=1)],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.segmentation),
        )


@pytest.fixture
def fxt_layout_dataset():
    return Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=1,
                            id=0,
                            group=0,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                            }, 
                        ),
                        Bbox(5.5, 6.0, 2.0, 2.0, label=2, group=0),
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.person_layout),
        )

@pytest.fixture
def fxt_action_dataset():
    return Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=1,
                            id=0,
                            group=0,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                                **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                            },
                        )
                    ],
                ),
                DatasetItem(
                    id="2007_000002",
                    subset="test",
                    media=Image.from_numpy(data=np.ones((10, 20, 3))),
                ),
            ],
            categories=VOC.make_voc_categories(task=VocTask.action_classification),
    )

class VocFormatTest(TestDataFormatBase):
    @pytest.mark.parametrize(
        "fxt_dataset_dir, importer",
        [
            (DUMMY_DATASET_DIR, VocClassificationImporter),
            (DUMMY_DATASET_DIR, VocDetectionImporter),
            (DUMMY_DATASET_DIR, VocSegmentationImporter),
            (DUMMY_DATASET_DIR, VocLayoutImporter),
            (DUMMY_DATASET_DIR, VocActionImporter),
        ],
        ids=["cls", "det", "seg", "layout", "action"],
    )
    def test_can_detect(self, fxt_dataset_dir: str, importer: Importer):
        detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(fxt_dataset_dir)
        assert importer.NAME in detected_formats

    @pytest.mark.parametrize(
        [
            "fxt_dataset_dir",
            "fxt_expected_dataset",
            "importer",
            "fxt_import_kwargs",
        ],
        [
            (DUMMY_DATASET_DIR, "fxt_classification_dataset", VocClassificationImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_detection_dataset", VocDetectionImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_segmentation_dataset", VocSegmentationImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_layout_dataset", VocLayoutImporter, {}),
            (DUMMY_DATASET_DIR, "fxt_action_dataset", VocActionImporter, {}),
        ],
        indirect=["fxt_expected_dataset"],
        ids=["cls", "det", "seg", "layout", "action"],
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
            importer=importer,
        )

    @pytest.mark.parametrize(
        "fxt_expected_dataset, exporter, fxt_export_kwargs, importer, fxt_import_kwargs, ",
        [
            ("fxt_classification_dataset", VocClassificationExporter, {"label_map": "voc_classification"}, VocClassificationImporter, {}),
            ("fxt_detection_dataset", VocDetectionExporter, {"label_map": "voc_detection"}, VocDetectionImporter, {}),
            ("fxt_segmentation_dataset", VocSegmentationExporter, {"label_map": "voc_segmentation"}, VocSegmentationImporter, {}),
            ("fxt_layout_dataset", VocLayoutExporter, {"label_map": "voc_layout"}, VocLayoutImporter, {}),
            ("fxt_action_dataset", VocActionExporter, {"label_map": "voc_action"}, VocActionImporter, {}),
        ],
        indirect=["fxt_expected_dataset"],
    )
    def test_can_export_and_import(
        self,
        fxt_expected_dataset: Dataset,
        test_dir: str,
        fxt_import_kwargs: Dict[str, Any],
        fxt_export_kwargs: Dict[str, Any],
        request: pytest.FixtureRequest,
        exporter: Exporter,
        importer: Importer,
    ):
        return super().test_can_export_and_import(
            fxt_expected_dataset,
            test_dir,
            fxt_import_kwargs,
            fxt_export_kwargs,
            request,
            exporter=exporter,
            importer=importer,
        )

#     def test_can_import_voc_dataset_with_empty_lines_in_subset_lists(self):
#         expected_dataset = Dataset.from_iterable(
#             [
#                 DatasetItem(
#                     id="2007_000001",
#                     subset="train",
#                     media=Image.from_numpy(data=np.ones((10, 20, 3))),
#                     annotations=[
#                         Bbox(
#                             1.0,
#                             2.0,
#                             2.0,
#                             2.0,
#                             label=8,
#                             id=1,
#                             group=1,
#                             attributes={
#                                 "difficult": False,
#                                 "truncated": True,
#                                 "occluded": False,
#                                 "pose": "Unspecified",
#                             },
#                         )
#                     ],
#                 )
#             ],
#             categories=VOC.make_voc_categories(task=VocTask.detection),
#         )

#         rpath = osp.join("ImageSets", "Main", "train.txt")
#         matrix = [
#             ("voc_detection", "", ""),
#             ("voc_detection", "train", rpath),
#         ]
#         for format, subset, path in matrix:
#             with self.subTest(format=format, subset=subset, path=path):
#                 if subset:
#                     expected = expected_dataset.get_subset(subset)
#                 else:
#                     expected = expected_dataset

#                 actual = Dataset.import_from(osp.join(DUMMY_DATASET3_DIR, path), format)

#                 compare_datasets(self, expected, actual, require_media=True)

#     def test_can_pickle(self):
#         formats = [
#             "voc_classification",
#             "voc_detection",
#             "voc_action",
#             "voc_layout",
#             "voc_segmentation",
#         ]

#         for fmt in formats:
#             with self.subTest(fmt=fmt):
#                 source = Dataset.import_from(DUMMY_DATASET1_DIR, format=fmt)

#                 parsed = pickle.loads(pickle.dumps(source))  # nosec

#                 compare_datasets_strict(self, source, parsed)