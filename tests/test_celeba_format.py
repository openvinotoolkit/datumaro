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
    'celeba_dataset')

class CelebaImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_475)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='000001', subset='train',
                image=np.ones((3, 4, 3)),
                annotations=[Label(12),
                    Bbox(95, 71, 226, 313, label=12),
                    Points([165, 184, 244, 176, 196, 249, 194, 271, 266, 260], label=12)],
                attributes={'5_o_Clock_Shadow': False, 'Arched_Eyebrows': True,
                    'Attractive': True, 'Bags_Under_Eyes': False, 'Bald': False,
                    'Bangs': False, 'Big_Lips': False, 'Big_Nose': False, 'Black_Hair': False,
                    'Blond_Hair': False, 'Blurry': False, 'Brown_Hair': True,
                    'Bushy_Eyebrows': False, 'Chubby': False, 'Double_Chin': False,
                    'Eyeglasses': False, 'Goatee': False, 'Gray_Hair': False,
                    'Heavy_Makeup': True, 'High_Cheekbones': True, 'Male': False,
                    'Mouth_Slightly_Open': True, 'Mustache': False, 'Narrow_Eyes': False,
                    'No_Beard': True, 'Oval_Face': False, 'Pale_Skin': False, 'Pointy_Nose': True,
                    'Receding_Hairline': False, 'Rosy_Cheeks': False, 'Sideburns': False,
                    'Smiling': True, 'Straight_Hair': True, 'Wavy_Hair': False,
                    'Wearing_Earrings': True, 'Wearing_Hat': False, 'Wearing_Lipstick': True,
                    'Wearing_Necklace': False, 'Wearing_Necktie': False, 'Young': True}
            ),
            DatasetItem(id='000002', subset='train',
                image=np.ones((3, 4, 3)),
                annotations=[Label(5),
                    Bbox(72, 94, 221, 306, label=5),
                    Points([140, 204, 220, 204, 168, 254, 146, 289, 226, 289], label=5)]
            ),
            DatasetItem(id='000003', subset='val',
                image=np.ones((3, 4, 3)),
                annotations=[Label(2),
                    Bbox(216, 59, 91, 126, label=2),
                    Points([244, 104, 264, 105, 263, 121, 235, 134, 251, 140], label=2)]
            ),
            DatasetItem(id='000004', subset='test',
                image=np.ones((3, 4, 3)),
                annotations=[Label(10),
                    Bbox(622, 257, 564, 781, label=10),
                    Points([796, 539, 984, 539, 930, 687, 762, 756, 915, 756], label=10)]
            ),
            DatasetItem(id='000005', subset='test',
                image=np.ones((3, 4, 3)),
                annotations=[Label(7),
                    Bbox(236, 109, 120, 166, label=7),
                    Points([273, 169, 328, 161, 298, 172, 283, 208, 323, 207], label=7)]
            )
        ], categories=['class_%d' % i for i in range(13)])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'celeba')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_475)
    def test_can_detect(self):
        self.assertTrue(CelebaImporter.detect(DUMMY_DATASET_DIR))
