from unittest import TestCase

import numpy as np
import os.path as osp

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType
)
from datumaro.plugins.imagenet_txt_format import \
    ImagenetTxtConverter, ImagenetTxtImporter
from datumaro.util.test_utils import TestDir, compare_datasets


class ImagenetTxtFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                annotations=[Label(0)]
            ),
            DatasetItem(id='2', subset='train',
                annotations=[Label(0)]
            ),
            DatasetItem(id='3', subset='train', image=np.zeros((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='4', subset='train',
                annotations=[Label(1)]
            ),
            DatasetItem(id='5', subset='train', image=np.zeros((4, 8, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='6', subset='train',
                annotations=[Label(5)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetTxtConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet_txt')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_and_load_with_multiple_labels(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                annotations=[Label(1), Label(3)]
            ),
            DatasetItem(id='2', subset='train', image=np.zeros((8, 6, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='3', subset='train', image=np.zeros((2, 8, 3)),
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetTxtConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet_txt')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/c', image=np.zeros((8, 4, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetTxtConverter.convert(source_dataset, test_dir,
                save_images=True)

            parsed_dataset = Dataset.import_from(test_dir, 'imagenet_txt')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'imagenet_txt_dataset')

class ImagenetTxtImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train', image=np.zeros((8, 6, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='2', subset='train', image=np.zeros((2, 8, 3)),
                annotations=[Label(5)]
            ),
            DatasetItem(id='3', subset='train',
                annotations=[Label(3)]
            ),
            DatasetItem(id='4', subset='train',
                annotations=[Label(5)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_%s' % label for label in range(10)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'imagenet_txt')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    def test_can_detect_imagenet(self):
        self.assertTrue(ImagenetTxtImporter.detect(DUMMY_DATASET_DIR))
