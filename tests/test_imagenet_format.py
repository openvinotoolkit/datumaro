from unittest import TestCase

import numpy as np
import os.path as osp

from datumaro.components.project import Project, Dataset
from datumaro.components.extractor import (DatasetItem, Label,
    LabelCategories, AnnotationType
)
from datumaro.plugins.imagenet_format import ImagenetConverter
from datumaro.plugins.imagenet_format import ImagenetImporter
from datumaro.util.test_utils import TestDir, compare_datasets


class ImagenetFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0_1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='label_0_2', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='label_0_1', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),

            DatasetItem(id='label_1_1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(2)]
            ),
            DatasetItem(id='label_1_2', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[Label(2)]
            ),

            DatasetItem(id='label_5_1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(6)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = ImagenetImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_multiple_labels(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1), Label(3)]
            ),
            DatasetItem(id='label_0_1', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='2', subset='train',
                image=np.ones((10, 10, 3)),
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = ImagenetImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/c',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = ImagenetImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'imagenet_dataset')

class ImagenetImporterTets(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0_1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='label_0_2', subset='train',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='label_1_1', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[Label(2)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(10)),
        })

        dataset = Project.import_from(DUMMY_DATASET_DIR, 'imagenet').make_dataset()

        compare_datasets(self, expected_dataset, dataset)

    def test_can_detect_imagenet(self):
        self.assertTrue(ImagenetImporter.detect(DUMMY_DATASET_DIR))
