import pickle  # nosec import_pickle
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.imagenet import (
    ImagenetExporter,
    ImagenetImporter,
    ImagenetWithSubsetDirsImporter,
)

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_datasets_strict


class ImagenetFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0:1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1:2", media=Image(data=np.ones((10, 10, 3))), annotations=[Label(1)]
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
                    id="label_0:1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id="label_1:1", media=Image(data=np.ones((8, 8, 3))), annotations=[Label(1)]
                ),
                DatasetItem(id="no_label:2", media=Image(data=np.ones((8, 8, 3)))),
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
                    id="label_0:кириллица с пробелом",
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
                DatasetItem(id="no_label:a", media=Image(path="a.JPEG", data=np.zeros((4, 3, 3)))),
                DatasetItem(id="no_label:b", media=Image(path="b.bmp", data=np.zeros((3, 4, 3)))),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            ImagenetExporter.convert(dataset, test_dir, save_media=True)

            parsed_dataset = Dataset.import_from(test_dir, "imagenet")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)


class ImagenetImporterTest(TestCase):
    DUMMY_DATASET_DIR = get_test_asset_path("imagenet_dataset")
    FORMAT_NAME = "imagenet"
    IMPORTER_NAME = ImagenetImporter.NAME

    def _create_expected_dataset(self):
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id="label_0:label_0_1",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_0:label_0_2",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id="label_1:label_1_1",
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = self._create_expected_dataset()
        dataset = Dataset.import_from(self.DUMMY_DATASET_DIR, self.FORMAT_NAME)

        compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_imagenet(self):
        detected_formats = Environment().detect_dataset(self.DUMMY_DATASET_DIR)
        self.assertEqual([self.IMPORTER_NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        source = Dataset.import_from(self.DUMMY_DATASET_DIR, format=self.FORMAT_NAME)

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(self, source, parsed)


class ImagenetWithSubsetDirsImporterTest(ImagenetImporterTest):
    DUMMY_DATASET_DIR = get_test_asset_path("imagenet_subsets_dataset")
    FORMAT_NAME = "imagenet_with_subset_dirs"
    IMPORTER_NAME = ImagenetWithSubsetDirsImporter.NAME

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        dataset = Dataset.import_from(self.DUMMY_DATASET_DIR, "imagenet_with_subset_dirs")

        for subset_name, subset in dataset.subsets().items():
            expected_dataset = self._create_expected_dataset().transform(
                "map_subsets", mapping={"default": subset_name}
            )
            compare_datasets(self, expected_dataset, subset, require_media=True)
