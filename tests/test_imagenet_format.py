from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Label, LabelCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.imagenet_format import ImagenetConverter, ImagenetImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class ImagenetFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0/1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_1/2',
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_multiple_labels(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0), Label(1)]
            ),
            DatasetItem(id='2',
                image=np.ones((8, 8, 3))
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        excepted_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0/1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_1/1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='no_label/2',
                image=np.ones((8, 8, 3))
            ),
        ], categories=['label_0', 'label_1'])

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, excepted_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id="label_0/кириллица с пробелом",
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
        ], categories=['label_0'])

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='no_label/a', image=Image(path='a.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem(id='no_label/b', image=Image(path='b.bmp',
                data=np.zeros((3, 4, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            ImagenetConverter.convert(dataset, test_dir, save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet')

            compare_datasets(self, dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'imagenet_dataset')

class ImagenetImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0/label_0_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_0/label_0_2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_1/label_1_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            )
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'imagenet')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_imagenet(self):
        self.assertTrue(ImagenetImporter.detect(DUMMY_DATASET_DIR))
