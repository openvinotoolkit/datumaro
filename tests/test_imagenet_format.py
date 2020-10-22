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
            DatasetItem(id='label_0_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_0_2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_0_1',
                image=np.ones((10, 10, 3)),
                annotations=[Label(0)]
            ),

            DatasetItem(id='label_1_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='label_1_2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(1)]
            ),

            DatasetItem(id='label_2_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(2)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(3)),
        })

        with TestDir() as test_dir:
            ImagenetConverter.convert(source_dataset, test_dir, save_images=True)

            parsed_dataset = ImagenetImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'imagenet_dataset')

class ImagenetImporterTest(TestCase):
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_0_2',
                image=np.ones((10, 10, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='label_1_1',
                image=np.ones((8, 8, 3)),
                annotations=[Label(1)]
            ),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(2)),
        })

        dataset = Project.import_from(DUMMY_DATASET_DIR, 'imagenet').make_dataset()

        compare_datasets(self, expected_dataset, dataset)

    def test_can_detect_imagenet(self):
        self.assertTrue(ImagenetImporter.detect(DUMMY_DATASET_DIR))
