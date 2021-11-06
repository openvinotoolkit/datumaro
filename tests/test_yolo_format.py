from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.yolo_format.converter import YoloConverter
from datumaro.plugins.yolo_format.extractor import YoloImporter
from datumaro.util.image import save_image
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class YoloFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(0, 1, 2, 3, label=4),
                ]),
            DatasetItem(id=2, subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(3, 3, 2, 3, label=4),
                    Bbox(2, 1, 2, 3, label=4),
                ]),

            DatasetItem(id=3, subset='valid', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 5, 2, label=2),
                    Bbox(0, 2, 3, 2, label=5),
                    Bbox(0, 2, 4, 2, label=6),
                    Bbox(0, 7, 3, 2, label=7),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(i) for i in range(10)),
        })

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'yolo')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                image=Image(path='1.jpg', size=(10, 15)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(3, 3, 2, 3, label=4),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(i) for i in range(10)),
        })

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir)

            save_image(osp.join(test_dir, 'obj_train_data', '1.jpg'),
                np.ones((10, 15, 3))) # put the image for dataset
            parsed_dataset = Dataset.import_from(test_dir, 'yolo')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                image=Image(path='1.jpg', size=(10, 15)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(3, 3, 2, 3, label=4),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(i) for i in range(10)),
        })

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir)

            parsed_dataset = Dataset.import_from(test_dir, 'yolo',
                image_info={'1': (10, 15)})

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', subset='train',
                image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(0, 1, 2, 3, label=4),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(i) for i in range(10)),
        })

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'yolo')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                image=np.ones((4, 2, 3))),
            DatasetItem(id='subdir1/1', subset='train',
                image=np.ones((2, 6, 3))),
            DatasetItem(id='subdir2/1', subset='train',
                image=np.ones((5, 4, 3))),
        ], categories=[])

        for save_images in {True, False}:
            with self.subTest(save_images=save_images):
                with TestDir() as test_dir:
                    YoloConverter.convert(source_dataset, test_dir,
                        save_images=save_images)
                    parsed_dataset = Dataset.import_from(test_dir, 'yolo')

                    compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem('q/1', subset='train',
                image=Image(path='q/1.JPEG', data=np.zeros((4, 3, 3)))),
            DatasetItem('a/b/c/2', subset='valid',
                image=Image(path='a/b/c/2.bmp', data=np.zeros((3, 4, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            YoloConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'yolo')

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        expected = Dataset.from_iterable([
            DatasetItem(1, subset='train', image=np.ones((2, 4, 3))),
            DatasetItem(2, subset='train', image=np.ones((3, 2, 3))),
        ], categories=[])

        with TestDir() as path:
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='train', image=np.ones((2, 4, 3))),
                DatasetItem(2, subset='train',
                    image=Image(path='2.jpg', size=(3, 2))),
                DatasetItem(3, subset='valid', image=np.ones((2, 2, 3))),
            ], categories=[])
            dataset.export(path, 'yolo', save_images=True)

            dataset.put(DatasetItem(2, subset='train', image=np.ones((3, 2, 3))))
            dataset.remove(3, 'valid')
            dataset.save(save_images=True)

            self.assertEqual({'1.txt', '2.txt', '1.jpg', '2.jpg'},
                set(os.listdir(osp.join(path, 'obj_train_data'))))
            self.assertEqual(set(),
                set(os.listdir(osp.join(path, 'obj_valid_data'))))
            compare_datasets(self, expected, Dataset.import_from(path, 'yolo'),
                require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'yolo_dataset')

class YoloImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertIn(YoloImporter.NAME, detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id=1, subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(3, 3, 2, 3, label=4),
                ]),
        ], categories={
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(i) for i in range(10)),
        })

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'yolo')

        compare_datasets(self, expected_dataset, dataset)
