import numpy as np

from unittest import TestCase

from datumaro.components.project import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.image_dir import ImageDirConverter
from datumaro.util.test_utils import TestDir, test_save_and_load


class ImageDirFormatTest(TestCase):
    def test_can_load(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((10, 6, 3))),
            DatasetItem(id=2, image=np.ones((5, 4, 3))),
        ])

        with TestDir() as test_dir:
            test_save_and_load(self, dataset, ImageDirConverter.convert,
                test_dir, importer='image_dir')

    def test_relative_paths(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((4, 2, 3))),
            DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3))),
            DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3))),
        ])

        with TestDir() as test_dir:
            test_save_and_load(self, dataset, ImageDirConverter.convert,
                test_dir, importer='image_dir')

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', image=np.ones((4, 2, 3))),
        ])

        with TestDir() as test_dir:
            test_save_and_load(self, dataset, ImageDirConverter.convert,
                test_dir, importer='image_dir')
