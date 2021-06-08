import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import numpy as np
from datumaro.components.extractor import (AnnotationType, Bbox, DatasetItem,
    Extractor, LabelCategories, Mask)
from datumaro.components.dataset import Dataset
from datumaro.plugins.kitti_format.converter import (
    KittiConverter,
)
from datumaro.plugins.kitti_format.format import (KittiTask, KittiPath, KittiLabelMap,
    make_kitti_categories, make_kitti_detection_categories,
    parse_label_map, write_label_map,
)
from datumaro.plugins.kitti_format.importer import KittiImporter
from datumaro.util.image import Image
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'kitti_dataset')

class KittiFormatTest(TestCase):
    def test_can_write_and_parse_labelmap(self):
        src_label_map = KittiLabelMap

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, 'label_colors.txt')

            write_label_map(file_path, src_label_map)
            dst_label_map = parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

class KittiImportTest(TestCase):
    def test_can_import_segmentation(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='000030_10',
                subset='training',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), id=0, label=3,
                        attributes={'is_crowd': True}),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), id=1, label=27,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), id=2, label=27,
                        attributes={'is_crowd': False}),
                ]
            ),
            DatasetItem(id='000030_11',
                subset='training',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), id=1, label=31,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), id=1, label=12,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), id=0, label=3,
                        attributes={'is_crowd': True}),
                ]
            ),
        ], categories=make_kitti_categories())

        parsed_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'kitti_segmentation'), 'kitti')

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_import_detection(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='000030_10',
                subset='training',
                image=np.ones((10, 10, 3)),
                annotations=[
                    Bbox(0, 1, 2, 2, label=2, id=0,
                        attributes={'truncated': True, 'occluded': False}),
                    Bbox(0, 5, 1, 3, label=1, id=1,
                        attributes={'truncated': False, 'occluded': False}),
                ]),
            DatasetItem(id='000030_11',
                subset='training',
                image=np.ones((10, 10, 3)), annotations=[
                    Bbox(0, 0, 2, 2, label=1, id=0,
                        attributes={'truncated': True, 'occluded': True}),
                    Bbox(4, 4, 2, 2, label=1, id=1,
                        attributes={'truncated': False, 'occluded': False}),
                    Bbox(6, 6, 1, 3, label=1, id=2,
                        attributes={'truncated': False, 'occluded': True}),
                ]),
        ], categories=make_kitti_detection_categories())

        parsed_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_DIR, 'kitti_detection'), 'kitti')

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_detect_kitti(self):
        self.assertTrue(KittiImporter.detect(DUMMY_DATASET_DIR))


class TestExtractorBase(Extractor):
    def _label(self, kitti_label):
        return self.categories()[AnnotationType.label].find(kitti_label)[0]

    def categories(self):
        return make_kitti_categories()

class KittiConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='kitti',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    def test_can_save_kitti_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1_2', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=15, id=0,
                            attributes={'is_crowd': True}),
                    ]),
                    DatasetItem(id='3', subset='val',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5, id=0,
                            attributes={'is_crowd': True}),
                    ]),
                ])
        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_can_save_kitti_detection(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1_2', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 1, 2, 2, label=2, id=0,
                            attributes={'truncated': False, 'occluded': False}),
                    ]),
                    DatasetItem(id='1_3', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 0, 2, 2, label=1, id=0,
                            attributes={'truncated': True, 'occluded': False}),
                        Bbox(6, 2, 3, 4, label=1, id=1,
                            attributes={'truncated': False, 'occluded': True}),
                    ]),
                ])

            def categories(self):
                return make_kitti_detection_categories()

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert,
                save_images=True, tasks=KittiTask.detection), test_dir)

    def test_can_save_kitti_detection_no_attributes(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1/2', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 1, 2, 2, label=2, id=0,
                            attributes={'truncated': False, 'occluded': False}),
                    ]),
                    DatasetItem(id='1/3', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 0, 2, 2, label=1, id=0,
                            attributes={'truncated': True, 'occluded': False}),
                        Bbox(6, 2, 3, 4, label=1, id=1,
                            attributes={'truncated': False, 'occluded': True}),
                    ]),
                ])

            def categories(self):
                return make_kitti_detection_categories()

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1/2', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 1, 2, 2, label=2, id=0,
                            attributes={'truncated': False, 'occluded': False}),
                    ]),
                    DatasetItem(id='1/3', subset='test',
                        image=np.ones((10, 10, 3)), annotations=[
                        Bbox(0, 0, 2, 2, label=1, id=0,
                            attributes={'truncated': False, 'occluded': False}),
                        Bbox(6, 2, 3, 4, label=1, id=1,
                            attributes={'truncated': False, 'occluded': False}),
                    ]),
                ])

            def categories(self):
                return make_kitti_detection_categories()

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert,
                save_images=True, allow_attributes=False,
                tasks=KittiTask.detection), test_dir,
                target_dataset=DstExtractor())

    def test_can_save_kitti_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1_2', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=15, id=0,
                            attributes={'is_crowd': True}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True, apply_colormap=False), test_dir)

    def test_can_save_kitti_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1_2',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 1]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                    ]),

                    DatasetItem(id='1_3',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2, id=0,
                            attributes={'is_crowd': True}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_can_save_kitti_dataset_without_frame_and_sequence(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='data', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                    ]),
                ])
        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='кириллица с пробелом',
                       image=np.ones((1, 5, 3)), annotations=[
                         Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=0,
                             attributes={'is_crowd': True}),
                         Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                             attributes={'is_crowd': False}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_can_save_kitti_dataset_with_complex_id(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_can_save_with_no_masks(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='city_1_2', subset='test',
                        image=np.ones((2, 5, 3)),
                    ),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, label_map='kitti',
                save_images=True), test_dir)

    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1, id=1,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=2,
                        attributes={'is_crowd': False}),
                ])

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add('background')
                label_cat.add('Label_1')
                label_cat.add('label_2')
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]),
                        attributes={'is_crowd': False}, id=1,
                        label=self._label('Label_1')),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]),
                        attributes={'is_crowd': False}, id=2,
                        label=self._label('label_2')),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = None
                label_map['Label_1'] = None
                label_map['label_2'] = None
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(KittiConverter.convert, label_map='source',
                save_images=True), test_dir, target_dataset=DstExtractor())

    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]), label=1, id=1,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), label=2, id=2,
                        attributes={'is_crowd': False}),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = (0, 0, 0)
                label_map['label_1'] = (1, 2, 3)
                label_map['label_2'] = (3, 2, 1)
                return make_kitti_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]),
                        attributes={'is_crowd': False}, id=1,
                        label=self._label('label_1')),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]),
                        attributes={'is_crowd': False}, id=2,
                        label=self._label('label_2')),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = (0, 0, 0)
                label_map['label_1'] = (1, 2, 3)
                label_map['label_2'] = (3, 2, 1)
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(KittiConverter.convert, label_map='source',
                save_images=True), test_dir, target_dataset=DstExtractor())

    def test_can_save_and_load_image_with_arbitrary_extension(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='q/1', image=Image(path='q/1.JPEG',
                        data=np.zeros((4, 3, 3)))),

                    DatasetItem(id='a/b/c/2', image=Image(
                             path='a/b/c/2.bmp', data=np.ones((1, 5, 3))
                         ), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 1]]), label=1, id=0,
                            attributes={'is_crowd': True}),
                    ]),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['a'] = None
                label_map['b'] = None
                return make_kitti_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(KittiConverter.convert, save_images=True),
                test_dir, require_images=True)
