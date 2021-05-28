import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Label, LabelCategories)
from datumaro.components.dataset import Dataset
from datumaro.plugins.widerface_format import WiderFaceConverter, WiderFaceImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets
from tests.requirements import Requirements, mark_requirement


class WiderFaceFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(0, 1, 2, 3, label=0, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                    Label(1),
                ]
            ),
            DatasetItem(id='2', subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '1',
                        'occluded': '0', 'pose': '1', 'invalid': '0'}),
                    Bbox(3, 3, 2, 3, label=0, attributes={
                        'blur': '0', 'expression': '1', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                    Bbox(2, 1, 2, 3, label=0, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '0', 'invalid': '1'}),
                    Label(2),
                ]
            ),

            DatasetItem(id='3', subset='val', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1.1, 5.3, 2.1, label=0, attributes={
                        'blur': '2', 'expression': '1', 'illumination': '0',
                        'occluded': '0', 'pose': '1', 'invalid': '0'}),
                    Bbox(0, 2, 3, 2, label=0, attributes={
                        'occluded': 'False'}),
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(0, 7, 3, 2, label=0, attributes={
                        'blur': '2', 'expression': '1', 'illumination': '0',
                        'occluded': '0', 'pose': '1', 'invalid': '0'}),
                ]
            ),

            DatasetItem(id='4', subset='val', image=np.ones((8, 8, 3))),
        ], categories=['face', 'label_0', 'label_1'])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'wider_face')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=2),
                    Bbox(0, 1, 2, 3, label=1, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                ]
            ),
        ], categories=['face', 'label_0', 'label_1'])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'wider_face')

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='кириллица с пробелом', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 2, 3, label=0, attributes = {
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                ]
            ),
        ], categories=['face'])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'wider_face')

            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_non_widerface_attributes(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(0, 1, 2, 3, label=0, attributes={
                        'non-widerface attribute': '0',
                        'blur': 1, 'invalid': '1'}),
                    Bbox(1, 1, 2, 2, label=0, attributes={
                        'non-widerface attribute': '0'}),
                ]
            ),
        ], categories=['face'])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, label=0),
                    Bbox(0, 1, 2, 3, label=0, attributes={
                        'blur': '1', 'invalid': '1'}),
                    Bbox(1, 1, 2, 2, label=0),
                ]
            ),
        ], categories=['face'])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'wider_face')

            compare_datasets(self, target_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        dataset = Dataset.from_iterable([
            DatasetItem('q/1', image=Image(path='q/1.JPEG',
                data=np.zeros((4, 3, 3)))),
            DatasetItem('a/b/c/2', image=Image(path='a/b/c/2.bmp',
                data=np.zeros((3, 4, 3)))),
        ], categories=[])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(dataset, test_dir, save_images=True)
            parsed_dataset = Dataset.import_from(test_dir, 'wider_face')

            compare_datasets(self, dataset, parsed_dataset, require_images=True)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'widerface_dataset')

class WiderFaceImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        self.assertTrue(WiderFaceImporter.detect(DUMMY_DATASET_DIR))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0_Parade_image_01', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 2, 2, 2, attributes={
                        'blur': '0', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '0', 'invalid': '0'}),
                    Label(0),
                ]
            ),
            DatasetItem(id='1_Handshaking_image_02', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 1, 2, 2, attributes={
                        'blur': '0', 'expression': '0', 'illumination': '1',
                        'occluded': '0', 'pose': '0', 'invalid': '0'}),
                    Bbox(5, 1, 2, 2, attributes={
                        'blur': '0', 'expression': '0', 'illumination': '1',
                        'occluded': '0', 'pose': '0', 'invalid': '0'}),
                    Label(1),
                ]
            ),
            DatasetItem(id='0_Parade_image_03', subset='val',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0, 0, 1, 1, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                    Bbox(3, 2, 1, 2, attributes={
                        'blur': '0', 'expression': '0', 'illumination': '0',
                        'occluded': '1', 'pose': '0', 'invalid': '0'}),
                    Bbox(5, 6, 1, 1, attributes={
                        'blur': '2', 'expression': '0', 'illumination': '0',
                        'occluded': '0', 'pose': '2', 'invalid': '0'}),
                    Label(0),
                ]
            ),
        ], categories= ['Parade', 'Handshaking'])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'wider_face')

        compare_datasets(self, expected_dataset, dataset)
