from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox, Label, Points
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.celeba_format import CelebaImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'celeba_dataset', 'celeba')
DUMMY_ALIGN_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'celeba_dataset', 'align_celeba')

class CelebaImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_475)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='000001', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Label(12),
                    Bbox(95, 71, 226, 313, label=12),
                    Points([165, 184, 244, 176, 196, 249, 194, 271, 266, 260],
                        label=12)
                ],
                attributes={'5_o_Clock_Shadow': False, 'Arched_Eyebrows': True,
                    'Attractive': True, 'Bags_Under_Eyes': False, 'Bald': False,
                    'Bangs': False, 'Big_Lips': False, 'Big_Nose': False}
            ),
            DatasetItem(id='000002', subset='train',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Label(5),
                    Bbox(72, 94, 221, 306, label=5),
                    Points([140, 204, 220, 204, 168, 254, 146, 289, 226, 289],
                        label=5)
                ]
            ),
            DatasetItem(id='000003', subset='val',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Label(2),
                    Bbox(216, 59, 91, 126, label=2),
                    Points([244, 104, 264, 105, 263, 121, 235, 134, 251, 140],
                        label=2)
                ],
                attributes={'5_o_Clock_Shadow': False, 'Arched_Eyebrows': False,
                    'Attractive': False, 'Bags_Under_Eyes': True, 'Bald': False,
                    'Bangs': False, 'Big_Lips': False, 'Big_Nose': True}
            ),
            DatasetItem(id='000004', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Label(10),
                    Bbox(622, 257, 564, 781, label=10),
                    Points([796, 539, 984, 539, 930, 687, 762, 756, 915, 756],
                        label=10)
                    ]
            ),
            DatasetItem(id='000005', subset='test',
                image=np.ones((5, 5, 3)),
                annotations=[
                    Label(7),
                    Bbox(236, 109, 120, 166, label=7),
                    Points([273, 169, 328, 161, 298, 172, 283, 208, 323, 207],
                        label=7)
                ]
            )
        ], categories=['class-%d' % i for i in range(13)])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'celeba')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    def test_can_import_align_dataset(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='000001', subset='train',
                image=np.ones((3, 4, 3)),
                annotations=[
                    Label(12),
                    Points([69, 109, 106, 113, 77, 142, 73, 152, 108, 154],
                        label=12)
                ],
                attributes={'5_o_Clock_Shadow': False, 'Arched_Eyebrows': True,
                    'Attractive': True, 'Bags_Under_Eyes': False, 'Bald': False,
                    'Bangs': False, 'Big_Lips': False, 'Big_Nose': False}
            ),
            DatasetItem(id='000002', subset='train',
                image=np.ones((3, 4, 3)),
                annotations=[
                    Label(5),
                    Points([69, 110, 107, 112, 81, 135, 70, 151, 108, 153],
                        label=5)
                ]
            ),
            DatasetItem(id='000003', subset='val',
                image=np.ones((3, 4, 3)),
                annotations=[
                    Label(2),
                    Points([76, 112, 104, 106, 108, 128, 74, 156, 98, 158],
                        label=2)
                ],
                attributes={'5_o_Clock_Shadow': False, 'Arched_Eyebrows': False,
                    'Attractive': False, 'Bags_Under_Eyes': True, 'Bald': False,
                    'Bangs': False, 'Big_Lips': False, 'Big_Nose': True}
            ),
            DatasetItem(id='000004', subset='test',
                image=np.ones((3, 4, 3)),
                annotations=[
                    Label(10),
                    Points([72, 113, 108, 108, 101, 138, 71, 155, 101, 151],
                        label=10)
                ]
            ),
            DatasetItem(id='000005', subset='test',
                image=np.ones((3, 4, 3)),
                annotations=[
                    Label(7),
                    Points([66, 114, 112, 112, 86, 119, 71, 147, 104, 150],
                        label=7)
                    ]
            )
        ], categories=['class-%d' % i for i in range(13)])

        dataset = Dataset.import_from(DUMMY_ALIGN_DATASET_DIR, 'celeba')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect(self):
        self.assertTrue(CelebaImporter.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect_align_dataset(self):
        self.assertTrue(CelebaImporter.detect(DUMMY_ALIGN_DATASET_DIR))
