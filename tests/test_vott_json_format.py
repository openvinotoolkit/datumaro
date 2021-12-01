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
    'vott_json_dataset')

class VottJsonImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_475)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0d3de147fea94f1797f9d012acad9666',
                subset='train', image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(5, 10, 10, 2, label=0)
                ]
            ),
            DatasetItem(id='b482849bcc1cc84684805cfe02ecb30c',
                subset='train', image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(11.5, 12, 10.2, 20.5, label=0),
                    Bbox(11.5, 12, 10.2, 20.5, label=1),
                ]
            ),
            DatasetItem(id='50fef05a87e49b8d77ad8e01c5a86909',
                subset='train', image=np.ones((5, 5, 3)),
                annotations=[
                    Bbox(6.7, 10.3, 3.3, 4.7),
                    Bbox(13.7, 20.2, 31.9, 43.4, label=2),
                ]
            )
        ], categories=['animal', 'dog', 'person'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'vott_json')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(VottJsonImporter.NAME, detected_formats)
