import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (AnnotationType, DatasetItem, Label,
    LabelCategories)
from datumaro.plugins.mnist_format import MnistConverter, MnistImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

import pytest
from tests.requirements import Requirements, DatumaroComponent

@pytest.mark.components(DatumaroComponent.Datumaro)
class MnistFormatTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
            MnistConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)


    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
            MnistConverter.convert(source_dataset, test_dir, save_images=False)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load_with_different_image_size(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=0, image=np.ones((3, 4)),
                annotations=[Label(0)]
            ),
            DatasetItem(id=1, image=np.ones((2, 2)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
            MnistConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
            MnistConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load_empty_image(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, annotations=[Label(0)]),
            DatasetItem(id=1)
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            MnistConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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
            MnistConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'mnist')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mnist_dataset')

@pytest.mark.components(DatumaroComponent.Datumaro)
class MnistImporterTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
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

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'mnist')

        compare_datasets(self, expected_dataset, dataset)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_detect(self):
        self.assertTrue(MnistImporter.detect(DUMMY_DATASET_DIR))
