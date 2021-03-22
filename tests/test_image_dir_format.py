import numpy as np
import os
import os.path as osp

from unittest import TestCase

from datumaro.components.project import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.plugins.image_dir_format import ImageDirConverter
from datumaro.util import image
from datumaro.util.image import Image, save_image
from datumaro.util.test_utils import TestDir, compare_datasets, test_save_and_load


class ImageDirFormatTest(TestCase):
    def test_can_load(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((10, 6, 3))),
            DatasetItem(id=2, image=np.ones((5, 4, 3))),
        ])

        with TestDir() as test_dir:
            test_save_and_load(self, dataset, ImageDirConverter.convert,
                test_dir, importer='image_dir', require_images=True)

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

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem(id='a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((3, 4, 3)))),
        ])

        with TestDir() as test_dir:
            test_save_and_load(self, dataset, ImageDirConverter.convert,
                test_dir, importer='image_dir', require_images=True)

    def test_can_save_and_load_image_with_custom_extension(self):
        expected = Dataset.from_iterable([
            DatasetItem(id='a/3', image=Image(path='a/3.qq',
                data=np.zeros((3, 4, 3)))),
        ])

        with TestDir() as test_dir:
            image_path = osp.join(test_dir, 'a', '3.jpg')
            save_image(image_path, expected.get('a/3').image.data,
                create_dir=True)
            os.rename(image_path, osp.join(test_dir, 'a', '3.qq'))

            actual = Dataset.import_from(test_dir, 'image_dir', exts='qq')
            compare_datasets(self, expected, actual, require_images=True)