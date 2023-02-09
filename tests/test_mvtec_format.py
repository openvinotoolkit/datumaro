import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mvtec.importer import (
    MvtecClassificationImporter,
    MvtecDetectionImporter,
    MvtecImporter,
    MvtecSegmentationImporter
)
from datumaro.plugins.data_formats.mvtec.exporter import MvtecExporter
from datumaro.util.test_utils import TestDir, compare_datasets, compare_datasets_strict

from .requirements import Requirements, mark_requirement


class MVTecFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="good/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="bad/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MVTecExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "mvtec_classification")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_multiple_labels(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0), Label(1)]
                ),
                DatasetItem(id="2", media=Image(data=np.ones((8, 8, 3)))),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        excepted_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1/1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]
                ),
                DatasetItem(id="no_label/2", media=Image(data=np.ones((8, 8, 3)))),
            ],
            categories=["label_0", "label_1"],
        )

        with TestDir() as test_dir:
            MVTecExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "mvtec_classification")

            compare_datasets(self, excepted_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "mvtec_dataset", "category_0")


class MvtecImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_classification(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="bad/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="good/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["bad", "good"]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_classification")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_segmentation(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="bad/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=np.ones((8, 8, 3)),
                            label=0,
                        ),
                    ],
                ),
                DatasetItem(
                    id="good/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=np.zeros((8, 8, 3)),
                            label=1,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["bad", "good"]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_segmentation")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_detection(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="bad/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=np.ones((8, 8, 3)),
                            label=0,
                        ),
                    ],
                ),
                DatasetItem(
                    id="good/000",
                    subset="test",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=np.zeros((8, 8, 3)),
                            label=1,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["bad", "good"]),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_segmentation")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_mvtec(self):
        matrix = [
            (DUMMY_DATASET_DIR, MvtecClassificationImporter),
            (DUMMY_DATASET_DIR, MvtecSegmentationImporter),
            (DUMMY_DATASET_DIR, MvtecDetectionImporter),
        ]

        env = Environment()

        for path, subtask in matrix:
            with self.subTest(path=path, task=subtask):
                detected_formats = env.detect_dataset(path)
                self.assertIn(subtask.NAME, detected_formats)
