import os.path as osp
from functools import partial
from unittest import TestCase

import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem, Points
from datumaro.plugins.lfw_format import LfwConverter, LfwImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets


class LfwFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0/name0_0001',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name0/name0_0002'],
                    'negative_pairs': []
                }
            ),
            DatasetItem(id='name0/name0_0002',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name0/name0_0001'],
                    'negative_pairs': ['name1/name1_0001']
                }
            ),
            DatasetItem(id='name1/name1_0001',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name1/name1_0002'],
                    'negative_pairs': []
                }
            ),
            DatasetItem(id='name1/name1_0002',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name1/name1_0002'],
                    'negative_pairs': ['name0/name0_0001']
                }
            ),
        ])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_landmarks(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0/name0_0001',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name0/name0_0002'],
                    'negative_pairs': []
                },
                annotations=[
                    Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0]),
                ]
            ),
            DatasetItem(id='name0/name0_0002',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': [],
                    'negative_pairs': []
                },
                annotations=[
                    Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0]),
                ]
            ),
        ])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0/name0_0001',
                image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name0/name0_0002'],
                    'negative_pairs': []
                },
            ),
            DatasetItem(id='name0/name0_0002',
                image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': [],
                    'negative_pairs': []
                },
            ),
        ])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='name0/name0_0001', image=Image(
                path='name0/name0_0001.JPEG', data=np.zeros((4, 3, 3))),
                attributes={
                    'positive_pairs': [],
                    'negative_pairs': []
                },
            ),
            DatasetItem(id='name0/name0_0002', image=Image(
                path='name0/name0_0002.bmp', data=np.zeros((3, 4, 3))),
                attributes={
                    'positive_pairs': ['name0/name0_0001'],
                    'negative_pairs': []
                },
            ),
        ])

        with TestDir() as test_dir:
            LfwConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'lfw_dataset')

class LfwImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(LfwImporter.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='name0/name0_0001',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': [],
                    'negative_pairs': ['name1/name1_0001',
                        'name1/name1_0002']
                },
                annotations=[
                    Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0]),
                ]
            ),
            DatasetItem(id='name1/name1_0001',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': ['name1/name1_0002'],
                    'negative_pairs': []
                },
                annotations=[
                    Points([1, 6, 4, 6, 3, 3, 2, 1, 4, 1]),
                ]
            ),
            DatasetItem(id='name1/name1_0002',
                subset='test', image=np.ones((2, 5, 3)),
                attributes={
                    'positive_pairs': [],
                    'negative_pairs': []
                },
                annotations=[
                    Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0]),
                ]
            ),
        ])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'lfw')

        compare_datasets(self, expected_dataset, dataset)
