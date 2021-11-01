from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import (
    AnnotationType, Label, LabelCategories, Points, PointsCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.align_celeba_format import AlignCelebaImporter
from datumaro.util.test_utils import compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_ALIGN_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'align_celeba_dataset',)

class AlignCelebaImporterTest(TestCase):
    def test_can_import(self):
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
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                f'class-{i}' for i in range(13)),
            AnnotationType.points: PointsCategories.from_iterable(
                [(0, ['lefteye_x']), (1, ['lefteye_y']), (2, ['righteye_x']), (3, ['righteye_y']),
                 (4, ['nose_x']), (5, ['nose_y']), (6, ['leftmouth_x']), (7, ['leftmouth_y']),
                 (8, ['rightmouth_x']), (9, ['rightmouth_y'])])
        })

        dataset = Dataset.import_from(DUMMY_ALIGN_DATASET_DIR, 'align_celeba')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect_align_dataset(self):
        self.assertTrue(AlignCelebaImporter.detect(DUMMY_ALIGN_DATASET_DIR))
