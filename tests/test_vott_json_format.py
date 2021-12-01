from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.vott_json_format import VottJsonImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'vott_json_dataset', 'dataset')
DUMMY_DATASET_DIR_WITH_META_FILE = osp.join(osp.dirname(__file__),
    'assets', 'vott_json_dataset', 'dataset_with_meta_file')

class VottJsonImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_475)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img0001',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': '0d3de147f'},
                annotations=[
                    Bbox(5, 10, 10, 2, label=0,
                        attributes={'id': 'BsO3zj9bn'})
                ]
            ),
            DatasetItem(id='img0002',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': 'b482849bc'},
                annotations=[
                    Bbox(11.5, 12, 10.2, 20.5, label=0,
                        attributes={'id': 'mosw0b97K'}),
                    Bbox(11.5, 12, 10.2, 20.5, label=1,
                        attributes={'id': 'mosw0b97K'})
                ]
            ),
            DatasetItem(id='img0003',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': '50fef05a8'},
                annotations=[
                    Bbox(6.7, 10.3, 3.3, 4.7,
                        attributes={'id': '35t9mf-Zr'}),
                    Bbox(13.7, 20.2, 31.9, 43.4, label=2,
                        attributes={'id': 'sO4m1DtTZ'})
                ]
            )
        ], categories=['animal', 'dog', 'person'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'vott_json')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_import_with_meta_file(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img0001',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': '0d3de147f'},
                annotations=[
                    Bbox(5, 10, 10, 2, label=0,
                        attributes={'id': 'BsO3zj9bn'})
                ]
            ),
            DatasetItem(id='img0002',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': 'b482849bc'},
                annotations=[
                    Bbox(11.5, 12, 10.2, 20.5, label=1,
                        attributes={'id': 'mosw0b97K'})
                ]
            ),
            DatasetItem(id='img0003',
                subset='train', image=np.ones((5, 5, 3)),
                attributes={'id': '50fef05a8'},
                annotations=[
                    Bbox(6.7, 10.3, 3.3, 4.7,
                        attributes={'id': '35t9mf-Zr'}),
                    Bbox(13.7, 20.2, 31.9, 43.4, label=2,
                        attributes={'id': 'sO4m1DtTZ'})
                ]
            )
        ], categories=['animal', 'dog', 'person'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR_WITH_META_FILE, 'vott_json')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(VottJsonImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect_with_meta_file(self):
        detected_formats = \
            Environment().detect_dataset(DUMMY_DATASET_DIR_WITH_META_FILE)
        self.assertIn(VottJsonImporter.NAME, detected_formats)
