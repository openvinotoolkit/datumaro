import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.imagenet import ImagenetExporter, ImagenetImporter
from datumaro.util.test_utils import TestDir, compare_datasets, compare_datasets_strict

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path


class ImagenetFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1/2", media=Image(data=np.ones((10, 10, 3))), annotations=[Label(1)]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            ImagenetExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet")

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
            ImagenetExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet")

            compare_datasets(self, excepted_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/кириллица с пробелом",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
            ],
            categories=["label_0"],
        )

        with TestDir() as test_dir:
            ImagenetExporter.convert(source_dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="no_label/a", media=Image(path="a.JPEG", data=np.zeros((4, 3, 3)))),
                DatasetItem(id="no_label/b", media=Image(path="b.bmp", data=np.zeros((3, 4, 3)))),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            ImagenetExporter.convert(dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("imagenet_dataset")


class ImagenetImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0/label_0_1",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_0/label_0_2",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_1/label_1_1",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_" + str(label) for label in range(2)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "imagenet")

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_imagenet(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(ImagenetImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        source = Dataset.import_from(DUMMY_DATASET_DIR, format="imagenet")

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(self, source, parsed)
