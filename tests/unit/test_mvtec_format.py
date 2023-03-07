from unittest import TestCase

import numpy as np
import cv2

from datumaro.components.annotation import AnnotationType, Bbox, Label, LabelCategories, Mask
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mvtec.exporter import MvtecExporter
from datumaro.plugins.data_formats.mvtec.importer import (
    MvtecClassificationImporter,
    MvtecDetectionImporter,
    MvtecSegmentationImporter,
)
from datumaro.util.test_utils import TestDir, compare_datasets

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path


class MVTecFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_classification(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_classification")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_segmentation(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.zeros((8, 8), dtype=np.uint8), label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_segmentation")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_detection(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=0, y=0, w=3, h=3, label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=4, y=4, w=3, h=3, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mvtec_detection")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("mvtec_dataset", "category_0")


class MvtecImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_classification(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_classification")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_segmentation(self):
        mask_label_1 = np.pad(np.ones((4, 4), dtype=np.uint8), ((0, 4), (0, 4)), "constant")
        mask_label_2 = np.pad(np.ones((4, 4), dtype=np.uint8), ((4, 0), (4, 0)), "constant")

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Mask(
                            image=mask_label_1,
                            label=1,
                        )
                    ],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=mask_label_2, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_segmentation")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_detection(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(label=0)],
                ),
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=0, y=0, w=3, h=3, label=1)],
                ),
                DatasetItem(
                    id="label_2/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(x=4, y=4, w=3, h=3, label=2)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(3)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mvtec_detection")

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


class MVTecExporterTest(TestCase):

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_segmentation_masks_saved_as_binary_image(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Mask(image=np.ones((8, 8), dtype=np.uint8), label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            assert cv2.imread(test_dir + "/ground_truth/label_1/000_mask.png").max() == 255

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_detection_masks_saved_as_binary_image(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_1/000",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(0, 0, 8, 8, label=1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MvtecExporter.convert(source_dataset, test_dir, save_media=True)
            assert cv2.imread(test_dir + "/ground_truth/label_1/000_mask.png").max() == 255
