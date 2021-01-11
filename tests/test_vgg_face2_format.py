import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import Bbox, DatasetItem, Points
from datumaro.components.dataset import Dataset
from datumaro.plugins.vgg_face2_format import (VggFace2Converter,
    VggFace2Importer)
from datumaro.util.test_utils import TestDir, compare_datasets


class VggFace2FormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2),
                    Points([3.2, 3.12, 4.11, 3.2, 2.11,
                        2.5, 3.5, 2.11, 3.8, 2.13]),
                ]
            ),
            DatasetItem(id='2', subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Points([4.23, 4.32, 5.34, 4.45, 3.54,
                        3.56, 4.52, 3.51, 4.78, 3.34]),
                ]
            ),
            DatasetItem(id='3', subset='val', image=np.ones((8, 8, 3))),
            DatasetItem(id='4', subset='val', image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2),
                    Points([3.2, 3.12, 4.11, 3.2, 2.11,
                        2.5, 3.5, 2.11, 3.8, 2.13]),
                    Bbox(2, 2, 1, 2),
                    Points([2.787, 2.898, 2.965, 2.79, 2.8,
                        2.456, 2.81, 2.32, 2.89, 2.3]),
                ]
            ),
            DatasetItem(id='5', subset='val', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(2, 2, 2, 2),
                ]
            ),
        ], categories=[])

        with TestDir() as test_dir:
            VggFace2Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = VggFace2Importer()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2),
                    Points([4.23, 4.32, 5.34, 4.45, 3.54,
                        3.56, 4.52, 3.51, 4.78, 3.34]),
                ]
            ),
        ], categories=[])

        with TestDir() as test_dir:
            VggFace2Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = VggFace2Importer()(test_dir).make_dataset()

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
                    Bbox(2, 2, 1, 2),
                    Points([2.787, 2.898, 2.965, 2.79, 2.8,
                        2.456, 2.81, 2.32, 2.89, 2.3]),
                ]
            ),
            DatasetItem(id='n000002/0002_01', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 3, 1, 1),
                    Points([1.2, 3.8, 1.8, 3.82, 1.51,
                        3.634, 1.43, 3.34, 1.65, 3.32])
                ]
            ),
        ], categories=[])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'vgg_face2')

        compare_datasets(self, expected_dataset, dataset)
