import os.path as osp
from functools import partial

from unittest import TestCase

import numpy as np

from datumaro.components.extractor import (Bbox, Caption, DatasetItem, Mask,
    Polygon)
from datumaro.components.project import Dataset
from datumaro.plugins.icdar_format.converter import (
    IcdarConverter, IcdarTextLocalizationConverter,
    IcdarTextSegmentationConverter, IcdarWordRecognitionConverter)
from datumaro.plugins.icdar_format.extractor import IcdarImporter
from datumaro.plugins.icdar_format.format import IcdarTask
from datumaro.util.image import Image
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

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'word_recognition'), 'icdar')

        compare_datasets(self, expected_dataset, dataset)

    def test_can_import_bboxes(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='img_1', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Polygon([0, 0, 3, 1, 4, 6, 1, 7],
                        attributes={'text': 'FOOD'}),
                ]
            ),
            DatasetItem(id='img_2', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0, 0, 2, 3, attributes={'text': 'RED'}),
                    Bbox(3, 3, 2, 3, attributes={'text': 'LION'}),
                ]
            ),
        ])

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'text_localization'), 'icdar')

        compare_datasets(self, expected_dataset, dataset)

    def test_can_import_masks(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                image=np.ones((2, 5, 3)),
                annotations=[
                    Mask(group=0,
                        image=np.array([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),
                        attributes={ 'index': 0, 'color': '108 225 132',
                            'text': 'F', 'center': '0 1'
                        }),
                    Mask(group=1,
                        image=np.array([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]]),
                        attributes={ 'index': 1, 'color': '82 174 214',
                            'text': 'T', 'center': '1 3'
                        }),
                    Mask(group=1,
                        image=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]]),
                        attributes={ 'index': 2, 'color': '241 73 144',
                            'text': 'h', 'center': '1 4'
                        }),
                ]
            ),
        ])

        dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'text_segmentation'), 'icdar')

        compare_datasets(self, expected_dataset, dataset)

class IcdarConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='icdar',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

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
                IcdarWordRecognitionConverter.convert, test_dir)

    def test_can_save_and_load_bboxes(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Bbox(1, 3, 6, 10),
                    Bbox(0, 1, 3, 5, attributes={'text': 'word_0'}),
                ]),
            DatasetItem(id=2, subset='train',
                annotations=[
                    Polygon([0, 0, 3, 0, 4, 7, 1, 8],
                        attributes={'text': 'word_1'}),
                    Polygon([1, 2, 5, 3, 6, 8, 0, 7]),
                ]),
            DatasetItem(id=3, subset='train',
                annotations=[
                    Polygon([2, 2, 8, 3, 7, 10, 2, 9],
                        attributes={'text': 'word_2'}),
                    Bbox(0, 2, 5, 9, attributes={'text': 'word_3'}),
                ]),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextLocalizationConverter.convert, test_dir)

    def test_can_save_and_load_masks(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                annotations=[
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), group=1,
                        attributes={ 'index': 1, 'color': '82 174 214', 'text': 'j',
                            'center': '0 3' }),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), group=1,
                        attributes={ 'index': 0, 'color': '108 225 132', 'text': 'F',
                            'center': '0 1' }),
                ]),
            DatasetItem(id=2, subset='train',
                annotations=[
                    Mask(image=np.array([[0, 0, 0, 0, 0, 1]]), group=0,
                        attributes={ 'index': 3, 'color': '183 6 28', 'text': ' ',
                            'center': '0 5' }),
                    Mask(image=np.array([[1, 0, 0, 0, 0, 0]]), group=1,
                        attributes={ 'index': 0, 'color': '108 225 132', 'text': 'L',
                            'center': '0 0' }),
                    Mask(image=np.array([[0, 0, 0, 1, 1, 0]]), group=1,
                        attributes={ 'index': 1, 'color': '82 174 214', 'text': 'o',
                            'center': '0 3' }),
                    Mask(image=np.array([[0, 1, 1, 0, 0, 0]]), group=0,
                        attributes={ 'index': 2, 'color': '241 73 144', 'text': 'P',
                            'center': '0 1' }),
                ]),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextSegmentationConverter.convert, test_dir)

    def test_can_save_and_load_with_no_subsets(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 3, 5),
                ]),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextLocalizationConverter.convert, test_dir)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 3, 5),
                ]),
        ])

        with TestDir() as test_dir:
            self._test_save_and_load(expected_dataset,
                IcdarTextLocalizationConverter.convert, test_dir)

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((3, 4, 3)))),
        ])

        for task in [None] + list(IcdarTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(expected,
                    partial(IcdarConverter.convert, save_images=True,
                        tasks=task),
                    test_dir, require_images=True)