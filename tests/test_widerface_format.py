import os.path as osp
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import Bbox, DatasetItem
from datumaro.components.project import Dataset, Project
from datumaro.plugins.widerface_format import WiderFaceConverter, WiderFaceImporter
from datumaro.util.test_utils import TestDir, compare_datasets


class WiderFaceFormatTest(TestCase):
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2),
                    Bbox(0, 1, 2, 3, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 2, 'invalid': 0}),
                ]
            ),
            DatasetItem(id='2', subset='train', image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 1,
                        'occlusion': 0, 'pose': 1, 'invalid': 0}),
                    Bbox(3, 3, 2, 3, attributes = {
                        'blur': 0, 'expression': 1, 'illumination': 0,
                        'occlusion': 0, 'pose': 2, 'invalid': 0}),
                    Bbox(2, 1, 2, 3, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 0, 'invalid': 1}),
                ]
            ),

            DatasetItem(id='3', subset='val', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 1, 5, 2, attributes = {
                        'blur': 2, 'expression': 1, 'illumination': 0,
                        'occlusion': 0, 'pose': 1, 'invalid': 0}),
                    Bbox(0, 2, 3, 2),
                    Bbox(0, 2, 4, 2),
                    Bbox(0, 7, 3, 2, attributes = {
                        'blur': 2, 'expression': 1, 'illumination': 0,
                        'occlusion': 0, 'pose': 1, 'invalid': 0}),
                ]
            ),

            DatasetItem(id='4', subset='val', image=np.ones((8, 8, 3))),
        ])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = WiderFaceImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_save_dataset_with_no_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='a/b/1', image=np.ones((8, 8, 3)),
                annotations=[
                    Bbox(0, 2, 4, 2),
                    Bbox(0, 1, 2, 3, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 2, 'invalid': 0}),
                ]
            ),
        ])

        with TestDir() as test_dir:
            WiderFaceConverter.convert(source_dataset, test_dir, save_images=True)
            parsed_dataset = WiderFaceImporter()(test_dir).make_dataset()

            compare_datasets(self, source_dataset, parsed_dataset)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'widerface_dataset')

class WiderFaceImporterTest(TestCase):
    def test_can_detect(self):
        self.assertTrue(WiderFaceImporter.detect(DUMMY_DATASET_DIR))

    def test_can_import(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='0--Parade/0_Parade_image_01', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 2, 2, 2, attributes = {
                        'blur': 0, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 0, 'invalid': 0}),
                ]
            ),
            DatasetItem(id='1--Handshaking/1_Handshaking_image_02', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(1, 1, 2, 2, attributes = {
                        'blur': 0, 'expression': 0, 'illumination': 1,
                        'occlusion': 0, 'pose': 0, 'invalid': 0}),
                    Bbox(5, 1, 2, 2, attributes = {
                        'blur': 0, 'expression': 0, 'illumination': 1,
                        'occlusion': 0, 'pose': 0, 'invalid': 0}),
                ]
            ),
            DatasetItem(id='0--Parade/0_Parade_image_03', subset='val',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0, 0, 1, 1, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 2, 'invalid': 0}),
                    Bbox(3, 2, 1, 2, attributes = {
                        'blur': 0, 'expression': 0, 'illumination': 0,
                        'occlusion': 1, 'pose': 0, 'invalid': 0}),
                    Bbox(5, 6, 1, 1, attributes = {
                        'blur': 2, 'expression': 0, 'illumination': 0,
                        'occlusion': 0, 'pose': 2, 'invalid': 0}),
                ]
            ),
        ])

        dataset = Project.import_from(DUMMY_DATASET_DIR, 'wider_face') \
            .make_dataset()

        compare_datasets(self, expected_dataset, dataset)
