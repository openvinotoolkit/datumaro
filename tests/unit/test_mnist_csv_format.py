import os
import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import AnnotationType, Label, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.mnist_csv import MnistCsvExporter, MnistCsvImporter
from datumaro.util import dump_json_file
from datumaro.util.meta_file_util import get_hashkey_file

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets, compare_hashkey_meta


class MnistCsvFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(0)],
                ),
                DatasetItem(id=1, subset="test", media=Image.from_numpy(data=np.ones((28, 28)))),
                DatasetItem(
                    id=2,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_saving_images(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id=0, subset="train", annotations=[Label(0)]),
                DatasetItem(id=1, subset="train", annotations=[Label(1)]),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(source_dataset, test_dir, save_media=False)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0, media=Image.from_numpy(data=np.ones((10, 8))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id=1, media=Image.from_numpy(data=np.ones((4, 3))), annotations=[Label(1)]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(0)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id="q/1", media=Image.from_numpy(data=np.zeros((28, 28)), ext=".JPEG")),
                DatasetItem(
                    id="a/b/c/2", media=Image.from_numpy(data=np.zeros((28, 28)), ext=".bmp")
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable(
            [DatasetItem(id=0, annotations=[Label(0)]), DatasetItem(id=1)],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_other_labels(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0, media=Image.from_numpy(data=np.ones((28, 28))), annotations=[Label(0)]
                ),
                DatasetItem(
                    id=1, media=Image.from_numpy(data=np.ones((28, 28))), annotations=[Label(1)]
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    "label_%s" % label for label in range(2)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            compare_datasets(self, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(0)],
                ),
                DatasetItem(id=1, subset="test", media=Image.from_numpy(data=np.ones((28, 28)))),
                DatasetItem(
                    id=2,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(1)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        with TestDir() as test_dir:
            MnistCsvExporter.convert(
                source_dataset, test_dir, save_media=True, save_dataset_meta=True
            )
            parsed_dataset = Dataset.import_from(test_dir, "mnist_csv")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


DUMMY_DATASET_DIR = get_test_asset_path("mnist_csv_dataset")


class MnistCsvImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id=1,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(2)],
                ),
                DatasetItem(
                    id=2,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(1)],
                ),
                DatasetItem(
                    id=0,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(5)],
                ),
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(7)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "mnist_csv")

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([MnistCsvImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_hash_key(self):
        hashkey_meta = {
            "hashkey": {
                "test/0": np.zeros((1, 64), dtype=np.uint8).tolist(),
                "train/1": np.ones((1, 64), dtype=np.uint8).tolist(),
            }
        }
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=0,
                    subset="test",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(0)],
                ),
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((28, 28))),
                    annotations=[Label(7)],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    str(label) for label in range(10)
                ),
            },
        )
        with TestDir() as test_dir:
            MnistCsvExporter.convert(source_dataset, test_dir, save_media=True)

            meta_file = get_hashkey_file(test_dir)
            os.makedirs(osp.join(test_dir, "hash_key_meta"))
            dump_json_file(meta_file, hashkey_meta, indent=True)

            imported_dataset = Dataset.import_from(test_dir, "mnist_csv")
            compare_hashkey_meta(self, hashkey_meta, imported_dataset)
