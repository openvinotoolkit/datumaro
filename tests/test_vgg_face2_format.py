import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import Bbox, DatasetItem, Label, Points
from datumaro.components.dataset import Dataset
from datumaro.plugins.vgg_face2_format import (VggFace2Converter,
    VggFace2Importer)
from datumaro.util.test_utils import TestDir, compare_datasets


class VggFace2FormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0/1', subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Points([3.2, 3.12, 4.11, 3.2, 2.11,
                        2.5, 3.5, 2.11, 3.8, 2.13], label=0),
                ]
            ),
            DatasetItem(id='label_1/2', subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Points([4.23, 4.32, 5.34, 4.45, 3.54,
                        3.56, 4.52, 3.51, 4.78, 3.34], label=1),
                ]
            ),
            DatasetItem(id='label_2/3', subset='train', image=np.ones((8, 8, 3)),
                annotations=[Label(2)]
            ),
            DatasetItem(id='label_3/4', subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=3),
                    Points([3.2, 3.12, 4.11, 3.2, 2.11,
                        2.5, 3.5, 2.11, 3.8, 2.13], label=3),
                ]
            ),
            DatasetItem(id='5', subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(2, 2, 2, 2),
                ]
            ),
            DatasetItem(id='6', subset='train', image=np.ones((8, 8, 3)),
            ),
        ], categories=['label_%s' % i for i in range(4)])

        with TestDir() as test_dir:
            VggFace2Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'vgg_face2')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Points([4.23, 4.32, 5.34, 4.45, 3.54,
                        3.56, 4.52, 3.51, 4.78, 3.34], label=0),
                ]
            ),
        ], categories=['a'])

        with TestDir() as test_dir:
            VggFace2Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'vgg_face2')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_save_images(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='label_0/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Points([4.23, 4.32, 5.34, 4.45, 3.54,
                        3.56, 4.52, 3.51, 4.78, 3.34], label=0),
                ]
            ),
        ], categories=['label_0'])

        with TestDir() as test_dir:
            VggFace2Converter.convert(source_dataset, test_dir, save_images=False)
            parsed_dataset = Dataset.import_from(test_dir, 'vgg_face2')

            compare_datasets(self, source_dataset, parsed_dataset)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'vgg_face2_dataset')

class VggFace2ImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(VggFace2Importer.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='n000001/0001_01', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(2, 2, 1, 2, label=0),
                    Points([2.787, 2.898, 2.965, 2.79, 2.8,
                        2.456, 2.81, 2.32, 2.89, 2.3], label=0),
                ]
            ),
            DatasetItem(id='n000002/0002_01', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 3, 1, 1, label=1),
                    Points([1.2, 3.8, 1.8, 3.82, 1.51,
                        3.634, 1.43, 3.34, 1.65, 3.32], label=1)
                ]
            ),
        ], categories=['n000001', 'n000002'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'vgg_face2')

        compare_datasets(self, expected_dataset, dataset)
