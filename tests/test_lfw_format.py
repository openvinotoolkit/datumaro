from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.annotation import Label, Points
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.lfw_format import LfwConverter, LfwImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class LfwFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0_0001', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0, attributes={
                    'positive_pairs': ['name0/name0_0002']
                })]
            ),
            DatasetItem(id='name0_0002', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0, attributes={
                    'positive_pairs': ['name0/name0_0001'],
                    'negative_pairs': ['name1/name1_0001']
                })]
            ),
            DatasetItem(id='name1_0001', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[Label(1, attributes={
                    'positive_pairs': ['name1/name1_0002']
                })]
            ),
            DatasetItem(id='name1_0002', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[Label(1, attributes={
                    'positive_pairs': ['name1/name1_0002'],
                    'negative_pairs': ['name0/name0_0001']
                })]
            ),
        ], categories=['name0', 'name1'])

        for save_images in (True, False):
            with self.subTest(save_images=save_images), TestDir() as test_dir:
                LfwConverter.convert(source_dataset, test_dir,
                    save_images=save_images)
                parsed_dataset = Dataset.import_from(test_dir, 'lfw')

                compare_datasets(self, source_dataset, parsed_dataset,
                    require_images=save_images)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_landmarks(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0_0001',
                subset='test', image=np.ones((2, 5, 3)),
                annotations=[
                    Label(0, attributes={
                        'positive_pairs': ['name0/name0_0002']
                    }),
                    Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0]),
                ]
            ),
            DatasetItem(id='name0_0002',
                subset='test', image=np.ones((2, 5, 3)),
                annotations=[
                    Label(0),
                    Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0]),
                ]
            ),
        ], categories=['name0'])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='name0_0001',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0, attributes={
                    'positive_pairs': ['name0/name0_0002']
                })],
            ),
            DatasetItem(id='name0_0002',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0)]
            ),
        ], categories=['name0'])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_no_format_names(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/1',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0, attributes={
                    'positive_pairs': ['name0/b/2'],
                    'negative_pairs': ['d/4']
                })],
            ),
            DatasetItem(id='b/2',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0)]
            ),
            DatasetItem(id='c/3',
                image=np.ones((2, 5, 3)),
                annotations=[Label(1)]
            ),
            DatasetItem(id='d/4',
                image=np.ones((2, 5, 3)),
            ),
        ], categories=['name0', 'name1'])

        with TestDir() as test_dir:
            LfwConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом',
                image=np.ones((2, 5, 3))
            ),
            DatasetItem(id='name0_0002',
                image=np.ones((2, 5, 3)),
                annotations=[Label(0, attributes={
                    'negative_pairs': ['кириллица с пробелом']
                })]
            ),
        ], categories=['name0'])

        with TestDir() as test_dir:
            LfwConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='a/1', image=Image(
                path='a/1.JPEG', data=np.zeros((4, 3, 3))),
            ),
            DatasetItem(id='b/c/d/2', image=Image(
                path='b/c/d/2.bmp', data=np.zeros((3, 4, 3))),
            ),
        ], categories=[])

        with TestDir() as test_dir:
            LfwConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'lfw')

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'lfw_dataset')

class LfwImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(LfwImporter.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='name0_0001', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[
                    Label(0, attributes={
                        'negative_pairs': ['name1/name1_0001',
                            'name1/name1_0002']
                    }),
                    Points([0, 4, 3, 3, 2, 2, 1, 0, 3, 0]),
                ]
            ),
            DatasetItem(id='name1_0001', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[
                    Label(1, attributes={
                        'positive_pairs': ['name1/name1_0002'],
                    }),
                    Points([1, 6, 4, 6, 3, 3, 2, 1, 4, 1]),
                ]
            ),
            DatasetItem(id='name1_0002', subset='test',
                image=np.ones((2, 5, 3)),
                annotations=[
                    Label(1),
                    Points([0, 5, 3, 5, 2, 2, 1, 0, 3, 0]),
                ]
            ),
        ], categories=['name0', 'name1'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'lfw')

        compare_datasets(self, expected_dataset, dataset)
