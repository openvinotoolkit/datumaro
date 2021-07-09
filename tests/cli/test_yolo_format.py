from unittest import TestCase
import os.path as osp

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (DatasetItem,
    AnnotationType, Bbox)
from datumaro.util.test_utils import TestDir, compare_datasets, run_datum as run
from datumaro.components.extractor import AnnotationType, Bbox, DatasetItem
from datumaro.util.test_utils import TestDir, compare_datasets
import datumaro.plugins.voc_format.format as VOC

from ..requirements import Requirements, mark_requirement


class YoloIntegrationScenarios(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_yolo_dataset(self):
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(3.0, 3.0, 2.0, 3.0, label=4),
                    Bbox(0.0, 2.0, 4.0, 2.0, label=2)
                ]
            )
        ], categories=['label_' + str(i) for i in range(10)])

        with TestDir() as test_dir:
            yolo_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
                'tests', 'assets', 'yolo_dataset')

            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-p', test_dir, '-f', 'yolo', yolo_dir)

            export_dir = osp.join(test_dir, 'export_dir')
            run(self, 'export', '-p', test_dir, '-o', export_dir,
                '-f', 'yolo', '--', '--save-images')

            parsed_dataset = Dataset.import_from(export_dir, format='yolo')
            compare_datasets(self, target_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_mot_as_yolo(self):
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                annotations=[
                    Bbox(0.0, 4.0, 4.0, 8.0, label=2)
                ]
            )
        ], categories=['label_' + str(i) for i in range(10)])

        with TestDir() as test_dir:
            mot_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
                'tests', 'assets', 'mot_dataset')

            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-p', test_dir, '-f', 'mot_seq', mot_dir)

            yolo_dir = osp.join(test_dir, 'yolo_dir')
            run(self, 'export', '-p', test_dir, '-o', yolo_dir,
                '-f', 'yolo', '--', '--save-images')

            parsed_dataset = Dataset.import_from(yolo_dir, format='yolo')
            compare_datasets(self, target_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_convert_voc_to_yolo(self):
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='2007_000001', subset='train',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Bbox(1.0, 2.0, 2.0, 2.0, label=8),
                    Bbox(4.0, 5.0, 2.0, 2.0, label=15),
                    Bbox(5.5, 6, 2, 2, label=22),
                ]
            )
        ], categories=[label.name for label in
            VOC.make_voc_categories()[AnnotationType.label]])

        with TestDir() as test_dir:
            voc_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
                'tests', 'assets', 'voc_dataset', 'voc_dataset1')
            yolo_dir = osp.join(test_dir, 'yolo_dir')

            run(self, 'convert', '-if', 'voc', '-i', voc_dir,
                '-f', 'yolo', '-o', yolo_dir, '--', '--save-images')

            parsed_dataset = Dataset.import_from(yolo_dir, format='yolo')
            compare_datasets(self, target_dataset, parsed_dataset,
                require_images=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_ignore_non_supported_subsets(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='img1', subset='test',
                image=np.ones((10, 20, 3)),
                annotations=[
                    Bbox(1.0, 2.0, 1.0, 1.0, label=0)
                ]
            ),
            DatasetItem(id='img2', subset='train',
                image=np.ones((10, 5, 3)),
                annotations=[
                    Bbox(3.0, 1.0, 2.0, 1.0, label=1)
                ]
            )
        ], categories=[str(i) for i in range(4)])

        target_dataset = Dataset.from_iterable([
            DatasetItem(id='img2', subset='train',
                image=np.ones((10, 5, 3)),
                annotations=[
                    Bbox(3.0, 1.0, 2.0, 1.0, label=1)
                ]
            )
        ], categories=[str(i) for i in range(4)])

        with TestDir() as test_dir:
            dataset_dir = osp.join(test_dir, 'dataset_dir')
            source_dataset.save(dataset_dir, save_images=True)

            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-p', test_dir, '-f', 'datumaro', dataset_dir)

            yolo_dir = osp.join(test_dir, 'yolo_dir')
            run(self, 'export', '-p', test_dir, '-o', yolo_dir,
                '-f', 'yolo', '--', '--save-images')

            parsed_dataset = Dataset.import_from(yolo_dir, format='yolo')
            compare_datasets(self, target_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_delete_labels_from_yolo_dataset(self):
        target_dataset = Dataset.from_iterable([
            DatasetItem(id='1', subset='train',
                image=np.ones((10, 15, 3)),
                annotations=[
                    Bbox(0.0, 2.0, 4.0, 2.0, label=0)
                ]
            )
        ], categories=['label_2'])

        with TestDir() as test_dir:
            yolo_dir = osp.join(__file__[:__file__.rfind(osp.join('tests', ''))],
                'tests', 'assets', 'yolo_dataset')

            run(self, 'create', '-o', test_dir)
            run(self, 'add', '-p', test_dir, '-f', 'yolo', yolo_dir)

            run(self, 'filter', '-p', test_dir,
                '-m', 'i+a', '-e', "/item/annotation[label='label_2']")

            run(self, 'transform', '-p', test_dir,
                '-t', 'remap_labels', '--', '-l', 'label_2:label_2',
                '--default', 'delete')

            export_dir = osp.join(test_dir, 'export')
            run(self, 'export', '-p', test_dir, '-o', export_dir,
               '-f', 'yolo', '--', '--save-image')

            parsed_dataset = Dataset.import_from(export_dir, format='yolo')
            compare_datasets(self, target_dataset, parsed_dataset)
