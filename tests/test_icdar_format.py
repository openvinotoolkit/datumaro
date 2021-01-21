import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import Bbox, Caption, DatasetItem, Points
from datumaro.components.project import Dataset
from datumaro.plugins.icdar_format.converter import (
    IcdarTextLocalizationConverter, IcdarWordRecognitionConverter)
from datumaro.plugins.icdar_format.extractor import IcdarImporter
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'icdar_dataset')

class IcdarImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(IcdarImporter.detect(
            osp.join(DUMMY_DATASET_DIR, 'word_recognition')))

    def test_can_import_captions(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='word_1', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Caption('PROPER'),
                ]
            ),
            DatasetItem(id='word_2', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Caption("Canon"),
                ]
            ),
        ])

        dataset = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, 'word_recognition'), 'icdar')

        compare_datasets(self, expected_dataset, dataset)

    def test_can_import_bboxes(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img_1', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Points([0, 0, 3, 1, 4, 6, 1, 7], label=0),
                ]
            ),
            DatasetItem(id='img_2', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0, 0, 2, 3, label=2),
                    Bbox(3, 3, 2, 3, label=1),
                ]
            ),
        ], categories=['FOOD', 'LION', 'RED'])

        dataset = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, 'text_localization'), 'icdar')

        compare_datasets(self, expected_dataset, dataset)

class IcdarConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='icdar',
            target_dataset=target_dataset, importer_args=importer_args)

    def test_can_save_and_load_captions(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Caption('caption_0'),
                ]),
            DatasetItem(id=2, subset='train',
                annotations=[
                    Caption('caption_1'),
                ]),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarWordRecognitionConverter.convert, osp.join(test_dir, 'word_recognition'))

    def test_can_save_and_load_bboxes(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Bbox(1, 3, 6, 10),
                    Bbox(0, 1, 3, 5, label=0),
                ]),
            DatasetItem(id=2, subset='train',
                annotations=[
                    Points([0, 0, 3, 0, 4, 7, 1, 8], label=2),
                    Points([1, 2, 5, 3, 6, 8, 0, 7]),
                ]),
            DatasetItem(id=3, subset='train',
                annotations=[
                    Points([2, 2, 8, 3, 7, 10, 2, 9], label=1),
                    Bbox(0, 2, 5, 9, label=0),
                ]),
        ], categories=['label_0', 'label_1', 'label_2'])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextLocalizationConverter.convert,
                osp.join(test_dir, 'text_localization'))

    def test_can_save_and_load_with_no_subsets(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 3, 5),
                ]),
        ], categories=[])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextLocalizationConverter.convert,
                osp.join(test_dir, 'text_localization'))
