import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import datumaro.plugins.cityscapes_format as Cityscapes
import numpy as np
from datumaro.components.extractor import (AnnotationType, DatasetItem,
    Extractor, LabelCategories, Mask)
from datumaro.components.dataset import Dataset
from datumaro.plugins.cityscapes_format import (CityscapesImporter,
    CityscapesConverter)
from datumaro.util.image import Image
from datumaro.util.test_utils import (TestDir, compare_datasets,
    check_save_and_load)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets',
    'cityscapes_dataset')

class CityscapesFormatTest(TestCase):
    def test_can_write_and_parse_labelmap(self):
        src_label_map = Cityscapes.CityscapesLabelMap

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, 'label_colors.txt')

            Cityscapes.write_label_map(file_path, src_label_map)
            dst_label_map = Cityscapes.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

class CityscapesImportTest(TestCase):
    def test_can_import(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='defaultcity/defaultcity_000001_000031',
                subset='test',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), id=3, label=3,
                        attributes={'is_crowd': True}),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), id=1, label=27,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), id=2, label=27,
                        attributes={'is_crowd': False}),
                ]
            ),
            DatasetItem(id='defaultcity/defaultcity_000001_000032',
                subset='test',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), id=1, label=31,
                        attributes={'is_crowd': False}),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), id=12, label=12,
                        attributes={'is_crowd': True}),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), id=3, label=3,
                        attributes={'is_crowd': True}),
                ]
            ),
            DatasetItem(id='defaultcity/defaultcity_000002_000045',
                subset='train',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 1]]), id=3, label=3,
                        attributes={'is_crowd': True}),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), id=1, label=24,
                        attributes={'is_crowd': False}),
                ]
            ),
            DatasetItem(id='defaultcity/defaultcity_000001_000019',
                subset = 'val',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 0, 0, 1, 1]]), id=3, label=3,
                        attributes={'is_crowd': True}),
                    Mask(image=np.array([[0, 1, 1, 0, 0]]), id=24, label=1,
                        attributes={'is_crowd': False}),
                ]
            ),
        ], categories=Cityscapes.make_cityscapes_categories())

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'cityscapes')

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_detect_cityscapes(self):
        self.assertTrue(CityscapesImporter.detect(DUMMY_DATASET_DIR))


class TestExtractorBase(Extractor):
    def _label(self, cityscapes_label):
        return self.categories()[AnnotationType.label].find(cityscapes_label)[0]

    def categories(self):
        return Cityscapes.make_cityscapes_categories()

class CityscapesConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None, **kwargs):
        return check_save_and_load(self, source_dataset, converter, test_dir,
            importer='cityscapes',
            target_dataset=target_dataset, importer_args=importer_args, **kwargs)

    def test_can_save_cityscapes_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='defaultcity_1_2', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=15, id=15,
                            attributes={'is_crowd': True}),
                    ]),
                    DatasetItem(id='defaultcity_3', subset='val',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=5, id=5,
                            attributes={'is_crowd': True}),
                    ]),
                ])
        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
                save_images=True), test_dir)

    def test_can_save_cityscapes_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='defaultcity_1_2', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=15, id=15,
                            attributes={'is_crowd': True}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
                save_images=True, apply_colormap=False), test_dir)

    def test_can_save_cityscapes_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='defaultcity_1_2',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0, id=0,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 1]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                    ]),

                    DatasetItem(id='defaultcity_1_3',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1, id=1,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2, id=2,
                            attributes={'is_crowd': True}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
                save_images=True), test_dir)

    def test_can_save_cityscapes_dataset_without_frame_and_sequence(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='justcity', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                    ]),
                ])
        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
                save_images=True), test_dir)

    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='кириллица с пробелом',
                       image=np.ones((1, 5, 3)), annotations=[
                         Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=3,
                             attributes={'is_crowd': True}),
                         Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                             attributes={'is_crowd': False}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
                save_images=True), test_dir)

    def test_can_save_cityscapes_dataset_with_strange_id(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 1]]), label=3, id=3,
                            attributes={'is_crowd': True}),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=24, id=1,
                            attributes={'is_crowd': False}),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, label_map='cityscapes',
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
                partial(CityscapesConverter.convert, label_map='cityscapes',
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
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(CityscapesConverter.convert, label_map='source',
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
                return Cityscapes.make_cityscapes_categories(label_map)

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
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(CityscapesConverter.convert, label_map='source',
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
                        Mask(image=np.array([[0, 1, 1, 0, 1]]), label=1, id=1,
                            attributes={'is_crowd': True}),
                    ]),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['a'] = None
                label_map['b'] = None
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CityscapesConverter.convert, save_images=True),
                test_dir, require_images=True)
