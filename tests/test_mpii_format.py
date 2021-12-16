from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Points
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.mpii_format import MpiiImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'mpii_dataset')

class MpiiImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_580)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='015601864',
                subset='train', image=np.ones((5, 5, 3)),
                annotations=[
                    Points([620.0, 394.0, 616.0, 269.0, 573.0, 185.0, 647.0,
                        188.0, 661.0, 221.0, 656.0, 231.0, 610.0, 187.0,
                        647.0, 176.0, 637.02, 189.818, 695.98, 108.182,
                        606.0, 217.0, 553.0, 161.0, 601.0, 167.0, 692.0,
                        185.0, 693.0, 240.0, 688.0, 313.0],
                        [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        attributes={'center': [594.000,257.000], 'scale': 3.021})
                ]
            )
        ])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'mpii')

        compare_datasets(self, expected_dataset, dataset, require_images=True)


    @mark_requirement(Requirements.DATUM_580)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([MpiiImporter.NAME], detected_formats)
