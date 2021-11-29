from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Label, LabelCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.mnist_csv_format import (
    MnistCsvConverter, MnistCsvImporter,
)
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class MnistCsvFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, subset='test',
                image=np.ones((28, 28))
            ),
            DatasetItem(id=2, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(1)]
            )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_without_saving_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='train',
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, subset='train',
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(source_dataset, test_dir, save_images=False)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, image=np.ones((10, 8)),
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, image=np.ones((4, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="кириллица с пробелом",
                image=np.ones((28, 28)),
                annotations=[Label(0)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((28, 28)))),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((28, 28)))),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, annotations=[Label(0)]),
            DatasetItem(id=1)
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_other_labels(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, image=np.ones((28, 28)),
                annotations=[Label(0)]),
            DatasetItem(id=1, image=np.ones((28, 28)),
                annotations=[Label(1)])
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_%s' % label for label in range(2)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, subset='test',
                image=np.ones((28, 28))
            ),
            DatasetItem(id=2, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(1)]
            )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistCsvConverter.convert(source_dataset, test_dir, save_images=True,
                save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist_csv')

            self.assertTrue(osp.isfile(osp.join(test_dir, 'dataset_meta.json')))
            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mnist_csv_dataset')

class MnistCsvImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(2)]
            ),
            DatasetItem(id=2, subset='test',
                image=np.ones((28, 28)),
                annotations=[Label(1)]
            ),
            DatasetItem(id=0, subset='train',
                image=np.ones((28, 28)),
                annotations=[Label(5)]
            ),
            DatasetItem(id=1, subset='train',
                image=np.ones((28, 28)),
                annotations=[Label(7)]
            )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'mnist_csv')

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(MnistCsvImporter.NAME, detected_formats)
