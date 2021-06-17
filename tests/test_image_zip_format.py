import os.path as osp
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.extractor import DatasetItem
from datumaro.components.project import Dataset
from datumaro.plugins.image_zip_format import ImageZipPath, ImageZipConverter
from datumaro.util.image import Image, save_image
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class ImageZipConverterTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def _test_can_save_and_load(self, source_dataset, test_dir,
            **kwargs):
        archive_path = osp.join(test_dir, kwargs.get('name',
            ImageZipPath.DEFAULT_ARCHIVE_NAME))
        ImageZipConverter.convert(source_dataset, test_dir, **kwargs)
        parsed_dataset = Dataset.import_from(archive_path, 'image_zip')

        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((10, 6, 3))),
            DatasetItem(id='2', image=np.ones((5, 4, 3))),
        ])

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_with_custom_archive_name(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='img_1', image=np.ones((10, 10, 3))),
        ])

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir,
                name='my_archive.zip')

    @mark_requirement(Requirements.DATUM_267)
    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((10, 10, 3))),
            DatasetItem(id='a/2', image=np.ones((4, 5, 3))),
            DatasetItem(id='a/b/3', image=np.ones((20, 10, 3)))
        ])

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_custom_compresion_method(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((5, 5, 3))),
            DatasetItem(id='2', image=np.ones((4, 3, 3))),
        ])

        with TestDir() as test_dir:
            self._test_can_save_and_load(source_dataset, test_dir,
                compression='ZIP_DEFLATED')

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_with_arbitrary_extensions(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='subset/1',
                image=Image(data=np.ones((10, 10, 3)), path='subset/1.png')),
            DatasetItem(id='2',
                image=Image(data=np.ones((4, 5, 3)), path='2.jpg')),
        ])

        with TestDir() as test_dir:
            save_image(osp.join(test_dir, '2.jpg'),
                source_dataset.get('2').image.data)
            save_image(osp.join(test_dir, 'subset', '1.png'),
                source_dataset.get('subset/1').image.data,
                create_dir=True)

            self._test_can_save_and_load(source_dataset, test_dir)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'image_zip_dataset')

class ImageZipImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def test_can_import(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((10, 10, 3)))
        ])

        zip_path = osp.join(DUMMY_DATASET_DIR, '1.zip')
        parsed_dataset = Dataset.import_from(zip_path, format='image_zip')
        compare_datasets(self, source_dataset, parsed_dataset)


    @mark_requirement(Requirements.DATUM_267)
    def test_can_import_from_directory(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', image=np.ones((10, 10, 3))),
            DatasetItem(id='2', image=np.ones((5, 10, 3)))
        ])

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, format='image_zip')
        compare_datasets(self, source_dataset, parsed_dataset)