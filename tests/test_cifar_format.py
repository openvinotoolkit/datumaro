import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (AnnotationType, DatasetItem, Label,
    LabelCategories)
from datumaro.plugins.cifar_format import CifarConverter, CifarImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets


class CifarFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.ones((32, 32, 3))
            ),
            DatasetItem(id='image_4', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            )
        ], categories=['label_0', 'label_1'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_and_load_without_saving_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='b',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label' + str(label) for label in range(2)),
        })

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=False)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='image_1',
                image=np.ones((10, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label' + str(label) for label in range(2)),
        })

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=False)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="кириллица с пробелом",
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
        ], categories=['label_0'])

        with TestDir() as test_dir:
            CifarConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((32, 32, 3)))),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((32, 32, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            CifarConverter.convert(dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='a', annotations=[Label(0)]),
            DatasetItem(id='b'),
        ], categories=['label_0'])

        with TestDir() as test_dir:
            CifarConverter.convert(dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'cifar')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'cifar_dataset')

class CifarImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='image_1', subset='train_1',
                image=np.ones((32, 32, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='image_2', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='image_3', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(3)]
            ),
            DatasetItem(id='image_4', subset='test',
                image=np.ones((32, 32, 3)),
                annotations=[Label(2)]
            )
        ], categories=['airplane', 'automobile', 'bird', 'cat'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'cifar')

        compare_datasets(self, expected_dataset, dataset)

    def test_can_detect(self):
        self.assertTrue(CifarImporter.detect(DUMMY_DATASET_DIR))
