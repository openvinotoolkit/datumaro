from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.vott_csv_format import VottCsvImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'vott_csv_dataset', 'dataset')
DUMMY_DATASET_DIR_WITH_META_FILE = osp.join(osp.dirname(__file__),
    'assets', 'vott_csv_dataset', 'dataset_with_meta_file')

class VottCsvImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_475)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img0001', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(10, 5, 10, 2, label=0)
                ]
            ),
            DatasetItem(id='img0002', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(11.5, 12, 10.2, 20.5, label=1),
                ]
            ),
            DatasetItem(id='img0003', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(6.7, 10.3, 3.3, 4.7, label=0),
                    Bbox(13.7, 20.2, 31.9, 43.4, label=1),
                ]
            ),
            DatasetItem(id='img0004', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(1, 2, 1, 2, label=0),
                ]
            )
        ], categories=['helmet', 'person'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'vott_csv')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_import_with_meta_file(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img0001', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(10, 5, 10, 2, label=0)
                ]
            ),
            DatasetItem(id='img0002', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(11.5, 12, 10.2, 20.5, label=1),
                ]
            )
        ], categories=['helmet', 'person'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_WITH_META_FILE,
            'vott_csv')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(VottCsvImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect_with_meta_file(self):
        detected_formats = \
            Environment().detect_dataset(DUMMY_DATASET_DIR_WITH_META_FILE)
        self.assertIn(VottCsvImporter.NAME, detected_formats)
