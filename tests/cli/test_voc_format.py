import os.path as osp
import numpy as np
from collections import OrderedDict

from unittest import TestCase
import pytest

import datumaro.plugins.voc_format.format as VOC
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.extractor import Bbox, Mask, Image, Label, LabelCategories, AnnotationType
from datumaro.cli.__main__ import main
from datumaro.util.test_utils import TestDir, compare_datasets
from ..requirements import Requirements, mark_requirement

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
        """
        <b>Description:</b>
        Testing a particular example of working with VOC dataset.

        <b>Expected results:</b>
        A VOC dataset that matches the expected result.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Create a datumaro project and add source dataset to it.
        3. Leave only non-occluded annotations with `filter` command.
        4. Split the dataset into subsets with `transform` command.
        5. Export the project to a VOC dataset with `export` command.
        6. Verify that the resulting dataset is equal to the expected result.
        """

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
    def test_export_to_format(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be exported from datumaro project
        to VOC format with command `datum export`.

        <b>Expected results:</b>
        A VOC dataset that matches the source dataset.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Create a datumaro project and add source dataset to it.
        3. Export the dataset to VOC format, using the `export` command.
        4. Verify that the resulting dataset is equal to the source dataset.
        """

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

    @pytest.mark.reqids(Requirements.DATUM_283)
    def test_convert_to_voc_format(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be converted to VOC format with
        command `datum convert`.

        <b>Expected results:</b>
        A VOC dataset that matches the source dataset.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Convert source dataset to VOC format, using the `convert` command.
        3. Verify that resulting dataset is equal to the source dataset.
        """

        label_map = OrderedDict(('label_' + str(i), [None, [], []]) for i in range(10))
        label_map['background'] = [None, [], []]
        label_map.move_to_end('background', last=False)

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='default',
                annotations=[
                    Bbox(0.0, 4.0, 4.0, 8.0,
                        attributes={
                            'difficult': False,
                            'truncated': False,
                            'occluded': False,
                            'visibility': '1.0',
                            'ignored': 'False'
                        },
                        id=1, label=3, group=1
                    )
                ]
            )
        ], categories=VOC.make_voc_categories(label_map))

        mot_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
            'tests', 'assets', 'mot_dataset')
        with TestDir() as test_dir:
            voc_dir = osp.join(test_dir, 'voc')
            run(self, 'convert', '-if', 'mot_seq', '-i', mot_dir,
                '-f', 'voc', '-o', voc_dir)

            target_dataset = Dataset.import_from(voc_dir, format='voc')
            compare_datasets(self, source_dataset, target_dataset)

    @pytest.mark.reqids(Requirements.DATUM_283)
    def test_convert_from_voc_format(self):
        """
        <b>Description:</b>
        Ensure that the dataset can be converted from VOC format with
        command `datum convert`.

        <b>Expected results:</b>
        A LabelMe dataset that matches the source dataset.

        <b>Steps:</b>
        1. Get path to the source dataset from assets.
        2. Convert source dataset to LabelMe format, using the `convert` command.
        3. Verify that resulting dataset is equal to the source dataset.
        """

        source_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='default',
                image=Image(path='2007_000001.jpg', size=(10, 20)),
                annotations=[Label(i) for i in range(11)]
            ),
            DatasetItem(id='2007_000002', subset='default',
               image=np.ones((10, 20, 3))
            )
        ], categories={AnnotationType.label: LabelCategories.from_iterable(
                sorted([l.name for l in VOC.VocLabel if l.value % 2 == 1])),
        })

        voc_dir = osp.join(DUMMY_DATASETS_DIR, 'voc_dataset1')
        with TestDir() as test_dir:
            cvat_dir = osp.join(test_dir, 'cvat')
            run(self, 'convert', '-if', 'voc', '-i', voc_dir,
                '-f', 'imagenet', '-o', cvat_dir)

            target_dataset = Dataset.import_from(cvat_dir, format='imagenet')
            compare_datasets(self, source_dataset, target_dataset)

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
