import os.path as osp
from collections import OrderedDict

import numpy as np
from unittest import TestCase

import datumaro.plugins.voc_format.format as VOC
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.extractor import Bbox, Mask, Label
from datumaro.cli.__main__ import main
from datumaro.util.test_utils import TestDir, compare_datasets
from ..requirements import Requirements, mark_requirement

DUMMY_DATASETS_DIR = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
    'tests', 'assets', 'voc_dataset')

def run(test, *args, expected_code=0):
    test.assertEqual(expected_code, main(args), str(args))


class VocIntegrationScenarios(TestCase):
    def _test_can_save_and_load(self, project_path, source_path, expected_dataset,
            dataset_format, result_path='', label_map=None):
        run(self, 'create', '-o', project_path)
        run(self, 'add', '-p', project_path, '-f', dataset_format, source_path)

        result_dir = osp.join(project_path, 'result')
        extra_args = ['--', '--save-images']
        if label_map:
            extra_args += ['--label-map', label_map]
        run(self, 'export', '-f', dataset_format, '-p', project_path,
            '-o', result_dir, *extra_args)

        result_path = osp.join(result_dir, result_path)
        parsed_dataset = Dataset.import_from(result_path, dataset_format)
        compare_datasets(self, expected_dataset, parsed_dataset,
            require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_preparing_dataset_for_train_model(self):
        target_dataset = Dataset.from_iterable([
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
            run(self, 'add', '-p', test_dir, '-f', 'voc', dataset_path)

            run(self, 'filter', '-p', test_dir, '-m', 'i+a',
                '-e', "/item/annotation[occluded='False']")

            run(self, 'transform', '-p', test_dir,
                '-t', 'random_split', '--', '-s', 'test:.5',
                '-s', 'train:.5', '--seed', '1')

            export_path = osp.join(test_dir, 'dataset')
            run(self, 'export', '-p', test_dir, '-f', 'voc',
                '-o', export_path, '--', '--label-map', 'voc')

            parsed_dataset = Dataset.import_from(export_path, format='voc')
            compare_datasets(self, target_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_convert_to_voc_format(self):
        label_map = OrderedDict(('label_%s' % i, [None, [], []]) for i in range(10))
        label_map['background'] = [None, [], []]
        label_map.move_to_end('background', last=False)

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train', image=np.ones((10, 15, 3)),
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
            run(self, 'add', '-p', test_dir, '-f', 'yolo', yolo_dir)

            voc_export = osp.join(test_dir, 'voc_export')
            run(self, 'export', '-p', test_dir, '-f', 'voc',
                '-o', voc_export, '--', '--save-images')

            parsed_dataset = Dataset.import_from(voc_export, format='voc')
            compare_datasets(self, source_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[Label(i) for i in range(22) if i % 2 == 1] + [
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15, id=1, group=1,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        },
                    ),
                    Bbox(1.0, 2.0, 2.0, 2.0, label=8, id=2, group=2,
                        attributes={
                            'difficult': False,
                            'truncated': True,
                            'occluded': False,
                            'pose': 'Unspecified'
                        }
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=22, id=0, group=1),
                    Mask(image=np.ones([10, 20]), label=2, group=1),
                ]),

            DatasetItem(id='2007_000002', subset='test',
               image=np.ones((10, 20, 3)))
        ], categories=VOC.make_voc_categories())

        voc_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        with TestDir() as test_dir:
            self._test_can_save_and_load(test_dir, voc_dir, source_dataset,
                'voc', label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_layout_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15, id=1, group=1,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        }
                    ),
                    Bbox(5.5, 6.0, 2.0, 2.0, label=22, id=0, group=1),
                ]),

            DatasetItem(id='2007_000002', subset='test',
                image=np.ones((10, 20, 3))),
        ], categories=VOC.make_voc_categories())

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        rpath = osp.join('ImageSets', 'Layout', 'train.txt')
        matrix = [
            ('voc_layout', '', ''),
            ('voc_layout', 'train', rpath),
            ('voc', 'train', rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    source = source_dataset.get_subset(subset)
                else:
                    source = source_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(test_dir,
                        osp.join(dataset_dir, path), source,
                        format, result_path=path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_classification_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[Label(i) for i in range(22) if i % 2 == 1]),

            DatasetItem(id='2007_000002', subset='test',
                image=np.ones((10, 20, 3))),
        ], categories=VOC.make_voc_categories())

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        rpath = osp.join('ImageSets', 'Main', 'train.txt')
        matrix = [
            ('voc_classification', '', ''),
            ('voc_classification', 'train', rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    source = source_dataset.get_subset(subset)
                else:
                    source = source_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(test_dir,
                        osp.join(dataset_dir, path), source,
                        format, result_path=path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_detection_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15, id=2, group=2,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        }
                    ),
                    Bbox(1.0, 2.0, 2.0, 2.0, label=8, id=1, group=1,
                        attributes={
                            'difficult': False,
                            'truncated': True,
                            'occluded': False,
                            'pose': 'Unspecified'
                        }
                    )
                ]),

            DatasetItem(id='2007_000002', subset='test',
                image=np.ones((10, 20, 3))),
        ], categories=VOC.make_voc_categories())

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        rpath = osp.join('ImageSets', 'Main', 'train.txt')
        matrix = [
            ('voc_detection', '', ''),
            ('voc_detection', 'train', rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    source = source_dataset.get_subset(subset)
                else:
                    source = source_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(test_dir,
                        osp.join(dataset_dir, path), source,
                        format, result_path=path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_segmentation_dataset(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Mask(image=np.ones([10, 20]), label=2, group=1)
                ]),

            DatasetItem(id='2007_000002', subset='test',
                image=np.ones((10, 20, 3))),
        ], categories=VOC.make_voc_categories())

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        rpath = osp.join('ImageSets', 'Segmentation', 'train.txt')
        matrix = [
            ('voc_segmentation', '', ''),
            ('voc_segmentation', 'train', rpath),
            ('voc', 'train', rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    source = source_dataset.get_subset(subset)
                else:
                    source = source_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(test_dir,
                        osp.join(dataset_dir, path), source,
                        format, result_path=path, label_map='voc')

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_voc_action_dataset(self):
        expected_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15, id=1, group=1,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            **{
                                a.name : a.value % 2 == 1
                                for a in VOC.VocAction
                            }
                        }
                    )
                ]),

            DatasetItem(id='2007_000002', subset='test',
                image=np.ones((10, 20, 3))),
        ], categories=VOC.make_voc_categories())

        dataset_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        rpath = osp.join('ImageSets', 'Action', 'train.txt')
        matrix = [
            ('voc_action', '', ''),
            ('voc_action', 'train', rpath),
            ('voc', 'train', rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                with TestDir() as test_dir:
                    self._test_can_save_and_load(test_dir,
                        osp.join(dataset_dir, path), expected,
                        format, result_path=path, label_map='voc')
