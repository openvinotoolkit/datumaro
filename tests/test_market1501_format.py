import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.market1501_format import (Market1501Converter,
    Market1501Importer)
from datumaro.util.test_utils import TestDir, compare_datasets


class Market1501FormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0001_c2s3_000001_00',
                subset='test', image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 1,
                    'person_id': 1,
                    'query': True
                }
            ),
            DatasetItem(id='0002_c4s2_000002_00',
                subset='test', image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 3,
                    'person_id': 2,
                    'query': False
                }
            ),
            DatasetItem(id='0001_c1s1_000003_00',
                subset='test', image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 0,
                    'person_id': 1,
                    'query': False
                }
            ),
        ])

        with TestDir() as test_dir:
            Market1501Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'market1501')

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0001_c2s3_000001_00',
                image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 1,
                    'person_id': 1,
                    'query': True
                }
            ),
        ])

        with TestDir() as test_dir:
            Market1501Converter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'market1501')

            compare_datasets(self, source_dataset, parsed_dataset)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'market1501_dataset')

class Market1501ImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(Market1501Importer.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0001_c2s3_000111_00',
                subset='test', image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 1,
                    'person_id': 1,
                    'query': True
                }
            ),
            DatasetItem(id='0001_c1s1_001051_00',
                subset='test', image=np.ones((2, 5, 3)),
                attributes = {
                    'camera_id': 0,
                    'person_id': 1,
                    'query': False
                }
            ),
        ])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'market1501')

        compare_datasets(self, expected_dataset, dataset)
