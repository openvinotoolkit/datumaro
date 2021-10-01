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
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='000001', subset='train',
                image=np.ones((3, 4, 3)),
                annotations=[Label(12),
                    Bbox(95, 71, 226, 313, label=12),
                    Points([165, 184, 244, 176, 196, 249, 194, 271, 266, 260], label=12)],
                attributes={'5_o_Clock_Shadow': '-1', 'Arched_Eyebrows': '1',
                    'Attractive': '1', 'Bags_Under_Eyes': '-1', 'Bald': '-1',
                    'Bangs': '-1', 'Big_Lips': '-1', 'Big_Nose': '-1', 'Black_Hair': '-1',
                    'Blond_Hair': '-1', 'Blurry': '-1', 'Brown_Hair': '1',
                    'Bushy_Eyebrows': '-1', 'Chubby': '-1', 'Double_Chin': '-1',
                    'Eyeglasses': '-1', 'Goatee': '-1', 'Gray_Hair': '-1',
                    'Heavy_Makeup': '1', 'High_Cheekbones': '1', 'Male': '-1',
                    'Mouth_Slightly_Open': '1', 'Mustache': '-1', 'Narrow_Eyes': '-1',
                    'No_Beard': '1', 'Oval_Face': '-1', 'Pale_Skin': '-1', 'Pointy_Nose': '1',
                    'Receding_Hairline': '-1', 'Rosy_Cheeks': '-1', 'Sideburns': '-1',
                    'Smiling': '1', 'Straight_Hair': '1', 'Wavy_Hair': '-1',
                    'Wearing_Earrings': '1', 'Wearing_Hat': '-1', 'Wearing_Lipstick': '1',
                    'Wearing_Necklace': '-1', 'Wearing_Necktie': '-1', 'Young': '1'}
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
        ], categories=['label_%d' % i for i in range(13)])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'celeba')

        compare_datasets(self, expected_dataset, dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(CelebaImporter.detect(DUMMY_DATASET_DIR))
