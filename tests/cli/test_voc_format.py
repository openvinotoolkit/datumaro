import os.path as osp
import numpy as np
from collections import OrderedDict

from unittest import TestCase

import datumaro.plugins.voc_format.format as VOC
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.extractor import Bbox, Mask, Image, Label
from datumaro.cli.__main__ import main
from datumaro.util.test_utils import TestDir, compare_datasets
from tests.requirements import Requirements, mark_requirement

DUMMY_DATASETS_DIR = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'voc_dataset')

def run(test, *args, expected_code=0):
    test.assertEqual(expected_code, main(args), str(args))


class VocIntegrationScenarios(TestCase):
    def _test_can_save_and_load(self, project_path, source_path, source_dataset,
            dataset_format, result_path=None, label_map=None):
        run(self, 'create', '-o', project_path)
        run(self, 'add', 'path', '-p', project_path, '-f', dataset_format, source_path)

        result_dir = osp.join(project_path, 'voc_dataset')
        run(self, 'export', '-f', dataset_format, '-p', project_path,
            '-o', result_dir, '--', '--label-map', label_map)

        result_path = osp.join(result_dir, result_path) if result_path else result_dir
        target_dataset = Dataset.import_from(result_path, dataset_format)
        compare_datasets(self, source_dataset, target_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preparing_dataset_for_train_model(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='c', subset='train',
                annotations=[
                    Bbox(3.0, 1.0, 8.0, 5.0,
                        attributes={
                            'truncated': False,
                            'occluded': False,
                            'difficult': False
                        },
                        id=1, label=2, group=1
                    )
                ]
            ),
            DatasetItem(id='d', subset='test',
                annotations=[
                    Bbox(4.0, 4.0, 4.0, 4.0,
                        attributes={
                            'truncated': False,
                            'occluded': False,
                            'difficult': False
                        },
                        id=1, label=3, group=1
                    )
                ]
            )
        ], categories=VOC.make_voc_categories())

        dataset_path = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset2')

        with TestDir() as test_dir:
            run(self, 'create', '-o', test_dir)
            run(self, 'add', 'path', '-p', test_dir, '-f', 'voc', dataset_path)

            result_path = osp.join(test_dir, 'result')
            run(self, 'filter', '-p', test_dir, '-m', 'i+a',
                '-e', "/item/annotation[occluded='False']", '-o', result_path)

            split_path = osp.join(test_dir, 'split')
            run(self, 'transform', '-p', result_path, '-o', split_path,
                '-t', 'random_split', '--', '-s', 'test:.5',
                '-s', 'train:.5', '--seed', '1')

            export_path = osp.join(test_dir, 'dataset')
            run(self, 'export', '-p', split_path, '-f', 'voc',
                '-o', export_path, '--', '--label-map', 'voc')

            parsed_dataset = Dataset.import_from(export_path, format='voc')
            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_convert_to_voc_format(self):
        label_map = OrderedDict(('label_' + str(i), [None, [], []]) for i in range(10))
        label_map['background'] = [None, [], []]
        label_map.move_to_end('background', last=False)

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                annotations=[
                    Bbox(0.0, 2.0, 4.0, 2.0,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False
                        },
                        id=1, label=3, group=1
                    ),
                    Bbox(3.0, 3.0, 2.0, 3.0,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False
                        },
                        id=2, label=5, group=2
                    )
                ]
            )
        ], categories=VOC.make_voc_categories(label_map))

        with TestDir() as test_dir:
            yolo_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
                'tests', 'assets', 'yolo_dataset')

            run(self, 'create', '-o', test_dir)
            run(self, 'add', 'path', '-p', test_dir, '-f', 'yolo', yolo_dir)

            voc_export = osp.join(test_dir, 'voc_export')
            run(self, 'export', '-p', test_dir, '-f', 'voc',
                '-o', voc_export)

            parsed_dataset = Dataset.import_from(voc_export, format='voc')
            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[Label(i) for i in range(22) if i % 2 == 1] + [
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        },
                        id=1, group=1
                    ),
                    Bbox(1.0, 2.0, 2.0, 2.0, label=8,
                        attributes={
                            'difficult': False,
                            'truncated': True,
                            'occluded': False,
                            'pose': 'Unspecified'
                        },
                        id=2, group=2
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=22,
                        id=0, group=1
                    ),
                    Mask(image=np.ones([5, 10]), label=2, group=1)
                ]
            ),
            DatasetItem(id='2007_000002', subset='test',
               image=np.ones((10, 20, 3))
            )
        ], categories=VOC.make_voc_categories())

        voc_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        with TestDir() as test_dir:
            self._test_can_save_and_load(test_dir, voc_dir, source_dataset,
                'voc', label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_layout_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        },
                        id=1, group=1
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=22,
                        id=0, group=1
                    ),
                ]
            ),
        ], categories=VOC.make_voc_categories())

        voc_layout_path = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1',
            'ImageSets', 'Layout', 'train.txt')

        with TestDir() as test_dir:
            result_voc_path = osp.join('ImageSets', 'Layout', 'train.txt')
            self._test_can_save_and_load(test_dir, voc_layout_path, source_dataset,
                'voc_layout', result_path=result_voc_path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_detect_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        },
                        id=2, group=2
                    ),
                    Bbox(1.0, 2.0, 2.0, 2.0, label=8,
                        attributes={
                            'difficult': False,
                            'truncated': True,
                            'occluded': False,
                            'pose': 'Unspecified'
                        },
                        id=1, group=1
                    )
                ]
            ),
        ], categories=VOC.make_voc_categories())

        voc_detection_path = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1',
            'ImageSets', 'Main', 'train.txt')

        with TestDir() as test_dir:
            result_voc_path = osp.join('ImageSets', 'Main', 'train.txt')
            self._test_can_save_and_load(test_dir, voc_detection_path, source_dataset,
                'voc_detection', result_path=result_voc_path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_segmentation_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[
                    Mask(image=np.ones([5, 10]), label=2, group=1)
                ]
            )
        ], categories=VOC.make_voc_categories())

        voc_segm_path = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1',
            'ImageSets', 'Segmentation', 'train.txt')

        with TestDir() as test_dir:
            result_voc_path = osp.join('ImageSets', 'Segmentation', 'train.txt')
            self._test_can_save_and_load(test_dir, voc_segm_path, source_dataset,
                'voc_segmentation', result_path=result_voc_path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_action_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        },
                        id=1, group=1
                    )
                ]
            )
        ], categories=VOC.make_voc_categories())

        voc_act_path = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1',
            'ImageSets', 'Action', 'train.txt')

        with TestDir() as test_dir:
            result_voc_path = osp.join('ImageSets', 'Action', 'train.txt')
            self._test_can_save_and_load(test_dir, voc_act_path, source_dataset,
                'voc_action', result_path=result_voc_path, label_map='voc')
