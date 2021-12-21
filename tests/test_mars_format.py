# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest.case import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.plugins.mars_format import MarsImporter
from datumaro.util.test_utils import compare_datasets

from tests.requirements import Requirements, mark_requirement

ASSETS_DIR = osp.join(osp.dirname(__file__), 'assets')
DUMMY_MARS_DATASET = osp.join(ASSETS_DIR, 'mars_dataset')

class MarsImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0001C1T0001F001', image=np.ones((10, 10, 3)),
                subset='train', annotations=[Bbox(0, 0, 10, 10, label=2)],
                attributes={'camera_id': 1, 'track_id': 1, 'frame_id': 1}
            ),
            DatasetItem(id='0000C6T0101F001', image=np.ones((10, 10, 3)),
                subset='train', annotations=[Bbox(0, 0, 10, 10, label=1)],
                attributes={'camera_id': 6, 'track_id': 101, 'frame_id': 1}
            ),
            DatasetItem(id='00-1C2T0081F201', image=np.ones((10, 10, 3)),
                subset='test', annotations=[Bbox(0, 0, 10, 10, label=0)],
                attributes={'camera_id': 2, 'track_id': 81, 'frame_id': 201}
            ),
        ], categories=['00-1', '0000', '0001'])

        imported_dataset = Dataset.import_from(DUMMY_MARS_DATASET, 'mars')
        compare_datasets(self, expected_dataset, imported_dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_MARS_DATASET)
        self.assertEqual([MarsImporter.NAME], detected_formats)
