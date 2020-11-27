import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import datumaro.plugins.camvid_format as Camvid
import numpy as np
from datumaro.components.extractor import (AnnotationType, DatasetItem,
                                           Extractor, LabelCategories, Mask)
from datumaro.components.project import Dataset, Project
from datumaro.plugins.camvid_format import CamvidConverter, CamvidImporter
from datumaro.util.test_utils import (TestDir, compare_datasets,
                                      test_save_and_load)


class CamvidFormatTest(TestCase):
    def test_can_write_and_parse_labelmap(self):
        src_label_map = Camvid.CamvidLabelMap

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, 'label_colors.txt')
            Camvid.write_label_map(file_path, src_label_map)
            dst_label_map = Camvid.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'camvid_dataset')

class TestExtractorBase(Extractor):
    def _label(self, camvid_label):
        return self.categories()[AnnotationType.label].find(camvid_label)[0]

    def categories(self):
        return Camvid.make_camvid_categories()

class CamvidImportTest(TestCase):
    def test_can_import(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id='0001TP_008550', subset='test',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=18),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=22),
                ]
            ),
            DatasetItem(id='0001TP_008580', subset='test',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 0, 0]]), label=2),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=27),
                ]
            ),
            DatasetItem(id='0001TP_006690', subset='train',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 1]]), label=3),
                    Mask(image=np.array([[0, 0, 1, 0, 0]]), label=18),
                ]
            ),
            DatasetItem(id='0016E5_07959', subset = 'val',
                image=np.ones((1, 5, 3)),
                annotations=[
                    Mask(image=np.array([[1, 1, 1, 0, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 0, 1, 1]]), label=8),
                ]
            ),
        ], categories=Camvid.make_camvid_categories())

        parsed_dataset = Project.import_from(DUMMY_DATASET_DIR, 'camvid').make_dataset()

        compare_datasets(self, source_dataset, parsed_dataset)

    def test_can_detect_camvid(self):
        self.assertTrue(CamvidImporter.detect(DUMMY_DATASET_DIR))

class CamvidConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='camvid',
            target_dataset=target_dataset, importer_args=importer_args)

    def test_can_save_camvid_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='test',
                        image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CamvidConverter.convert, label_map='camvid'),
                test_dir)

    def test_can_save_camvid_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=0),
                        Mask(image=np.array([[0, 1, 1, 0, 0]]), label=3),
                        Mask(image=np.array([[1, 0, 0, 0, 1]]), label=4),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CamvidConverter.convert,
                    label_map='camvid', apply_colormap=False),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 0, 0, 1, 0]]), label=0),
                        Mask(image=np.array([[0, 1, 1, 0, 1]]), label=3),
                    ]),

                    DatasetItem(id=2, image=np.ones((1, 5, 3)), annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CamvidConverter.convert, label_map='camvid'), test_dir)

    def test_can_save_with_no_masks(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='test',
                        image=np.ones((2, 5, 3)),
                    ),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(CamvidConverter.convert, label_map='camvid'),
                test_dir)

    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                ])

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add('Label_1')
                label_cat.add('label_2')
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 0]]), label=self._label('Label_1')),
                    Mask(image=np.array([[0, 0, 1, 0, 1]]), label=self._label('label_2')),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = None
                label_map['Label_1'] = None
                label_map['label_2'] = None
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(CamvidConverter.convert, label_map='source'),
                test_dir, target_dataset=DstExtractor())

    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 0]]), label=1),
                    Mask(image=np.array([[0, 0, 1, 0, 1]]), label=2),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = (0, 0, 0)
                label_map['label_1'] = (1, 2, 3)
                label_map['label_2'] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, image=np.ones((1, 5, 3)), annotations=[
                    Mask(image=np.array([[1, 1, 0, 1, 0]]), label=self._label('label_1')),
                    Mask(image=np.array([[0, 0, 1, 0, 1]]), label=self._label('label_2')),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = (0, 0, 0)
                label_map['label_1'] = (1, 2, 3)
                label_map['label_2'] = (3, 2, 1)
                return Camvid.make_camvid_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(CamvidConverter.convert, label_map='source'),
                test_dir, target_dataset=DstExtractor())
