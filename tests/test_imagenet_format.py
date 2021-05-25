from unittest import TestCase

import numpy as np
import os.path as osp

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType
)
from datumaro.plugins.imagenet_format import ImagenetConverter, ImagenetImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

import pytest
from tests.requirements import Requirements
from tests.requirements import DatumaroComponent

@pytest.mark.components(DatumaroComponent.Datumaro)
class ImagenetFormatTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load_with_multiple_labels(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0), Label(1), Label(2)]
            ),
            DatasetItem(id='2',
                image=np.ones((8, 8, 3))
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(3)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="кириллица с пробелом",
                image=np.ones((8, 8, 3)),
                annotations=[Label(0), Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='a', image=Image(path='a.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem(id='b', image=Image(path='b.bmp',
                data=np.zeros((3, 4, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            ImagenetConverter.convert(dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'imagenet_dataset')

@pytest.mark.components(DatumaroComponent.Datumaro)
class ImagenetImporterTest(TestCase):
    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0), Label(1)]
            ),
            DatasetItem(id='2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(0)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'imagenet')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @pytest.mark.priority_medium
    @pytest.mark.reqids(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.component
    def test_can_detect_imagenet(self):
        self.assertTrue(ImagenetImporter.detect(DUMMY_DATASET_DIR))
