from collections import OrderedDict
from functools import partial
import numpy as np
import os
import os.path as osp

from unittest import TestCase

from datumaro.components.extractor import (Extractor, DatasetItem,
    AnnotationType, Label, Bbox, Mask, LabelCategories,
)
import datumaro.plugins.voc_format.format as VOC
from datumaro.plugins.voc_format.converter import (
    VocConverter,
    VocClassificationConverter,
    VocDetectionConverter,
    VocLayoutConverter,
    VocActionConverter,
    VocSegmentationConverter,
)
from datumaro.plugins.voc_format.importer import VocImporter
from datumaro.components.dataset import Dataset
from datumaro.util.image import Image
from datumaro.util.test_utils import (TestDir, compare_datasets,
    test_save_and_load)


class VocFormatTest(TestCase):
    def test_colormap_generator(self):
        reference = np.array([
            [  0,   0,   0],
            [128,   0,   0],
            [  0, 128,   0],
            [128, 128,   0],
            [  0,   0, 128],
            [128,   0, 128],
            [  0, 128, 128],
            [128, 128, 128],
            [ 64,   0,   0],
            [192,   0,   0],
            [ 64, 128,   0],
            [192, 128,   0],
            [ 64,   0, 128],
            [192,   0, 128],
            [ 64, 128, 128],
            [192, 128, 128],
            [  0,  64,   0],
            [128,  64,   0],
            [  0, 192,   0],
            [128, 192,   0],
            [  0,  64, 128],
            [224, 224, 192], # ignored
        ])

        self.assertTrue(np.array_equal(reference, list(VOC.VocColormap.values())))

    def test_can_write_and_parse_labelmap(self):
        src_label_map = VOC.make_voc_label_map()
        src_label_map['qq'] = [None, ['part1', 'part2'], ['act1', 'act2']]
        src_label_map['ww'] = [(10, 20, 30), [], ['act3']]

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, 'test.txt')

            VOC.write_label_map(file_path, src_label_map)
            dst_label_map = VOC.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

class TestExtractorBase(Extractor):
    def _label(self, voc_label):
        return self.categories()[AnnotationType.label].find(voc_label)[0]

    def categories(self):
        return VOC.make_voc_categories()


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), 'assets', 'voc_dataset')

class VocImportTest(TestCase):
    def test_can_import(self):
        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='2007_000001', subset='train',
                        image=Image(path='2007_000001.jpg', size=(10, 20)),
                        annotations=[
                            Label(self._label(l.name))
                            for l in VOC.VocLabel if l.value % 2 == 1
                        ] + [
                            Bbox(1, 2, 2, 2, label=self._label('cat'),
                                attributes={
                                    'pose': VOC.VocPose(1).name,
                                    'truncated': True,
                                    'difficult': False,
                                    'occluded': False,
                                },
                                id=1, group=1,
                            ),
                            Bbox(4, 5, 2, 2, label=self._label('person'),
                                attributes={
                                    'truncated': False,
                                    'difficult': False,
                                    'occluded': False,
                                    **{
                                        a.name: a.value % 2 == 1
                                        for a in VOC.VocAction
                                    }
                                },
                                id=2, group=2,
                            ),
                            Bbox(5.5, 6, 2, 2, label=self._label(
                                    VOC.VocBodyPart(1).name),
                                group=2
                            ),
                            Mask(image=np.ones([5, 10]),
                                label=self._label(VOC.VocLabel(2).name),
                                group=1,
                            ),
                        ]
                    ),
                    DatasetItem(id='2007_000002', subset='test',
                        image=np.ones((10, 20, 3))),
                ])

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, 'voc')

        compare_datasets(self, DstExtractor(), dataset)

    def test_can_detect_voc(self):
        self.assertTrue(VocImporter.detect(DUMMY_DATASET_DIR))

class VocConverterTest(TestCase):
    def _test_save_and_load(self, source_dataset, converter, test_dir,
            target_dataset=None, importer_args=None):
        return test_save_and_load(self, source_dataset, converter, test_dir,
            importer='voc',
            target_dataset=target_dataset, importer_args=importer_args)

    def test_can_save_voc_cls(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/0', subset='a', annotations=[
                        Label(1),
                        Label(2),
                        Label(3),
                    ]),

                    DatasetItem(id=1, subset='b', annotations=[
                        Label(4),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocClassificationConverter.convert, label_map='voc'),
                test_dir)

    def test_can_save_voc_det(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/1', subset='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2,
                            attributes={ 'occluded': True }
                        ),
                        Bbox(2, 3, 4, 5, label=3,
                            attributes={ 'truncated': True },
                        ),
                    ]),

                    DatasetItem(id=2, subset='b', annotations=[
                        Bbox(5, 4, 6, 5, label=3,
                            attributes={ 'difficult': True },
                        ),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/1', subset='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2, id=1, group=1,
                            attributes={
                                'truncated': False,
                                'difficult': False,
                                'occluded': True,
                            }
                        ),
                        Bbox(2, 3, 4, 5, label=3, id=2, group=2,
                            attributes={
                                'truncated': True,
                                'difficult': False,
                                'occluded': False,
                            },
                        ),
                    ]),

                    DatasetItem(id=2, subset='b', annotations=[
                        Bbox(5, 4, 6, 5, label=3, id=1, group=1,
                            attributes={
                                'truncated': False,
                                'difficult': True,
                                'occluded': False,
                            },
                        ),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocDetectionConverter.convert, label_map='voc'),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_voc_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='a', annotations=[
                        # overlapping masks, the first should be truncated
                        # the second and third are different instances
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3,
                            z_order=3),
                        Mask(image=np.array([[0, 1, 1, 1, 0]]), label=4,
                            z_order=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3,
                            z_order=2),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='a', annotations=[
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4,
                            group=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3,
                            group=2),
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3,
                            group=3),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocSegmentationConverter.convert, label_map='voc'),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_voc_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', annotations=[
                        # overlapping masks, the first should be truncated
                        # the second and third are different instances
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3,
                            z_order=3),
                        Mask(image=np.array([[0, 1, 1, 1, 0]]), label=4,
                            z_order=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3,
                            z_order=2),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', annotations=[
                        Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4,
                            group=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3,
                            group=2),
                        Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3,
                            group=3),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocSegmentationConverter.convert,
                    label_map='voc', apply_colormap=False),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_voc_segm_with_many_instances(self):
        def bit(x, y, shape):
            mask = np.zeros(shape)
            mask[y, x] = 1
            return mask

        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', annotations=[
                        Mask(image=bit(x, y, shape=[10, 10]),
                            label=self._label(VOC.VocLabel(3).name),
                            z_order=10 * y + x + 1
                        )
                        for y in range(10) for x in range(10)
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', annotations=[
                        Mask(image=bit(x, y, shape=[10, 10]),
                            label=self._label(VOC.VocLabel(3).name),
                            group=10 * y + x + 1
                        )
                        for y in range(10) for x in range(10)
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocSegmentationConverter.convert, label_map='voc'),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_voc_layout(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2, id=1, group=1,
                            attributes={
                                'pose': VOC.VocPose(1).name,
                                'truncated': True,
                                'difficult': False,
                                'occluded': False,
                            }
                        ),
                        Bbox(2, 3, 1, 1, label=self._label(
                            VOC.VocBodyPart(1).name), group=1),
                        Bbox(5, 4, 3, 2, label=self._label(
                            VOC.VocBodyPart(2).name), group=1),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocLayoutConverter.convert, label_map='voc'), test_dir)

    def test_can_save_voc_action(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2,
                            attributes={
                                'truncated': True,
                                VOC.VocAction(1).name: True,
                                VOC.VocAction(2).name: True,
                            }
                        ),
                        Bbox(5, 4, 3, 2, label=self._label('person'),
                            attributes={
                                'truncated': True,
                                VOC.VocAction(1).name: True,
                                VOC.VocAction(2).name: True,
                            }
                        ),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a/b/1', subset='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2,
                            id=1, group=1, attributes={
                                'truncated': True,
                                'difficult': False,
                                'occluded': False,
                                # no attributes here in the label categories
                            }
                        ),
                        Bbox(5, 4, 3, 2, label=self._label('person'),
                            id=2, group=2, attributes={
                                'truncated': True,
                                'difficult': False,
                                'occluded': False,
                                VOC.VocAction(1).name: True,
                                VOC.VocAction(2).name: True,
                                **{
                                    a.name: False for a in VOC.VocAction
                                        if a.value not in {1, 2}
                                }
                            }
                        ),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocActionConverter.convert,
                    label_map='voc', allow_attributes=False), test_dir,
                target_dataset=DstExtractor())

    def test_can_save_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, annotations=[
                        Label(2),
                        Label(3),
                    ]),

                    DatasetItem(id=2, annotations=[
                        Label(3),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocConverter.convert, label_map='voc'), test_dir)

    def test_can_save_dataset_with_images(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='a', image=np.ones([4, 5, 3])),
                    DatasetItem(id=2, subset='a', image=np.ones([5, 4, 3])),

                    DatasetItem(id=3, subset='b', image=np.ones([2, 6, 3])),
                ])

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(TestExtractor(),
                    partial(VocConverter.convert, label_map='voc',
                        save_images=True, tasks=task),
                    test_dir)

    def test_dataset_with_voc_labelmap(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=self._label('cat'), id=1),
                    Bbox(1, 2, 3, 4, label=self._label('non_voc_label'), id=2),
                ])

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add(VOC.VocLabel.cat.name)
                label_cat.add('non_voc_label')
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    # drop non voc label
                    Bbox(2, 3, 4, 5, label=self._label('cat'), id=1, group=1,
                        attributes={
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                ])

            def categories(self):
                return VOC.make_voc_categories()

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(VocConverter.convert, label_map='voc'),
                test_dir, target_dataset=DstExtractor())

    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=0, id=1),
                    Bbox(1, 2, 3, 4, label=1, id=2),
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
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=self._label('Label_1'),
                        id=1, group=1, attributes={
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                    Bbox(1, 2, 3, 4, label=self._label('label_2'),
                        id=2, group=2, attributes={
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = [None, [], []]
                label_map['Label_1'] = [None, [], []]
                label_map['label_2'] = [None, [], []]
                return VOC.make_voc_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(VocConverter.convert, label_map='source'),
                test_dir, target_dataset=DstExtractor())

    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=0, id=1),
                    Bbox(1, 2, 3, 4, label=2, id=2),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['label_1'] = [(1, 2, 3), [], []]
                label_map['background'] = [(0, 0, 0), [], []] # can be not 0
                label_map['label_2'] = [(3, 2, 1), [], []]
                return VOC.make_voc_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=self._label('label_1'),
                        id=1, group=1, attributes={
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                    Bbox(1, 2, 3, 4, label=self._label('label_2'),
                        id=2, group=2, attributes={
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                ])

            def categories(self):
                label_map = OrderedDict()
                label_map['background'] = [(0, 0, 0), [], []]
                label_map['label_1'] = [(1, 2, 3), [], []]
                label_map['label_2'] = [(3, 2, 1), [], []]
                return VOC.make_voc_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(VocConverter.convert, label_map='source'),
                test_dir, target_dataset=DstExtractor())

    def test_dataset_with_fixed_labelmap(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(2, 3, 4, 5, label=self._label('foreign_label'), id=1),
                    Bbox(1, 2, 3, 4, label=self._label('label'), id=2, group=2,
                        attributes={'act1': True}),
                    Bbox(2, 3, 4, 5, label=self._label('label_part1'), group=2),
                    Bbox(2, 3, 4, 6, label=self._label('label_part2'), group=2),
                ])

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add('foreign_label')
                label_cat.add('label', attributes=['act1', 'act2'])
                label_cat.add('label_part1')
                label_cat.add('label_part2')
                return {
                    AnnotationType.label: label_cat,
                }

        label_map = OrderedDict([
            ('label', [None, ['label_part1', 'label_part2'], ['act1', 'act2']])
        ])

        dst_label_map = OrderedDict([
            ('background', [None, [], []]),
            ('label', [None, ['label_part1', 'label_part2'], ['act1', 'act2']])
        ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(id=1, annotations=[
                    Bbox(1, 2, 3, 4, label=self._label('label'), id=1, group=1,
                        attributes={
                            'act1': True,
                            'act2': False,
                            'truncated': False,
                            'difficult': False,
                            'occluded': False,
                        }
                    ),
                    Bbox(2, 3, 4, 5, label=self._label('label_part1'), group=1),
                    Bbox(2, 3, 4, 6, label=self._label('label_part2'), group=1),
                ])

            def categories(self):
                return VOC.make_voc_categories(dst_label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(SrcExtractor(),
                partial(VocConverter.convert, label_map=label_map),
                test_dir, target_dataset=DstExtractor())

    def test_can_save_dataset_with_image_info(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, image=Image(path='1.jpg', size=(10, 15))),
                ])

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(TestExtractor(),
                    partial(VocConverter.convert, label_map='voc', tasks=task),
                    test_dir)

    def test_relative_paths(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='1', image=np.ones((4, 2, 3))),
                    DatasetItem(id='subdir1/1', image=np.ones((2, 6, 3))),
                    DatasetItem(id='subdir2/1', image=np.ones((5, 4, 3))),
                ])

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(TestExtractor(),
                    partial(VocConverter.convert,
                        label_map='voc', save_images=True, tasks=task),
                    test_dir)

    def test_can_save_attributes(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2,
                            attributes={ 'occluded': True, 'x': 1, 'y': '2' }
                        ),
                    ]),
                ])

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter([
                    DatasetItem(id='a', annotations=[
                        Bbox(2, 3, 4, 5, label=2, id=1, group=1,
                            attributes={
                                'truncated': False,
                                'difficult': False,
                                'occluded': True,
                                'x': '1', 'y': '2', # can only read strings
                            }
                        ),
                    ]),
                ])

        with TestDir() as test_dir:
            self._test_save_and_load(TestExtractor(),
                partial(VocConverter.convert, label_map='voc'), test_dir,
                target_dataset=DstExtractor())

    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a',
                    annotations=[Bbox(0, 0, 0, 0, label=1)]),
                DatasetItem(2, subset='b',
                    annotations=[Bbox(0, 0, 0, 0, label=2)]),
                DatasetItem(3, subset='c', image=np.ones((2, 2, 3)),
                    annotations=[
                        Bbox(0, 0, 0, 0, label=3),
                        Mask(np.ones((2, 2)), label=1)
                    ]),
            ], categories=['a', 'b', 'c', 'd'])
            dataset.export(path, 'voc', save_images=True)
            os.unlink(osp.join(path, 'Annotations', '1.xml'))
            os.unlink(osp.join(path, 'Annotations', '2.xml'))
            os.unlink(osp.join(path, 'Annotations', '3.xml'))
            self.assertFalse(osp.isfile(osp.join(path, 'JPEGImages', '2.jpg')))
            self.assertTrue(osp.isfile(osp.join(path, 'JPEGImages', '3.jpg')))
            self.assertTrue(osp.isfile(
                osp.join(path, 'SegmentationObject', '3.png')))
            self.assertTrue(osp.isfile(
                osp.join(path, 'SegmentationClass', '3.png')))

            dataset.put(DatasetItem(2, subset='b', image=np.ones((3, 2, 3)),
                annotations=[Bbox(0, 0, 0, 0, label=3)]))
            dataset.remove(3, 'c')
            dataset.save(save_images=True)

            self.assertFalse(osp.isfile(osp.join(path, 'Annotations', '1.xml')))
            self.assertTrue(osp.isfile(osp.join(path, 'Annotations', '2.xml')))
            self.assertFalse(osp.isfile(osp.join(path, 'Annotations', '3.xml')))
            self.assertTrue(osp.isfile(osp.join(path, 'JPEGImages', '2.jpg')))
            self.assertFalse(osp.isfile(osp.join(path, 'JPEGImages', '3.jpg')))
            self.assertFalse(osp.isfile(
                osp.join(path, 'SegmentationObject', '3.png')))
            self.assertFalse(osp.isfile(
                osp.join(path, 'SegmentationClass', '3.png')))