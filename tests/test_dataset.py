from unittest import TestCase
import os
import os.path as osp

import numpy as np

from datumaro.components.dataset import (
    DEFAULT_FORMAT, Dataset, ItemStatus, eager_mode,
)
from datumaro.components.dataset_filter import (
    DatasetItemEncoder, XPathAnnotationsFilter, XPathDatasetFilter,
)
from datumaro.components.environment import Environment
from datumaro.components.errors import DatumaroError, RepeatedItemError
from datumaro.components.extractor import (
    DEFAULT_SUBSET_NAME, AnnotationType, Bbox, Caption, DatasetItem, Extractor,
    ItemTransform, Label, LabelCategories, Mask, Points, Polygon, PolyLine,
    Transform,
)
from datumaro.components.launcher import Launcher
from datumaro.util.image import Image
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


class DatasetTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_create_from_extractors(self):
        class SrcExtractor1(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='train', annotations=[
                        Bbox(1, 2, 3, 4),
                        Label(4),
                    ]),
                    DatasetItem(id=1, subset='val', annotations=[
                        Label(4),
                    ]),
                ])

        class SrcExtractor2(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='val', annotations=[
                        Label(5),
                    ]),
                ])

        class DstExtractor(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='train', annotations=[
                        Bbox(1, 2, 3, 4),
                        Label(4),
                    ]),
                    DatasetItem(id=1, subset='val', annotations=[
                        Label(4),
                        Label(5),
                    ]),
                ])

        dataset = Dataset.from_extractors(SrcExtractor1(), SrcExtractor2())

        compare_datasets(self, DstExtractor(), dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_from_iterable(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=1, subset='train', annotations=[
                        Bbox(1, 2, 3, 4, label=2),
                        Label(4),
                    ]),
                    DatasetItem(id=1, subset='val', annotations=[
                        Label(3),
                    ]),
                ])

            def categories(self):
                return { AnnotationType.label: LabelCategories.from_iterable(
                    ['a', 'b', 'c', 'd', 'e'])
                }

        actual = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', annotations=[
                Bbox(1, 2, 3, 4, label=2),
                Label(4),
            ]),
            DatasetItem(id=1, subset='val', annotations=[
                Label(3),
            ]),
        ], categories=['a', 'b', 'c', 'd', 'e'])

        compare_datasets(self, TestExtractor(), actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_datasets_with_empty_categories(self):
        expected = Dataset.from_iterable([
            DatasetItem(1, annotations=[
                Label(0),
                Bbox(1, 2, 3, 4),
                Caption('hello world'),
            ])
        ], categories=['a'])

        src1 = Dataset.from_iterable([
            DatasetItem(1, annotations=[ Bbox(1, 2, 3, 4, label=None) ])
        ], categories=[])

        src2 = Dataset.from_iterable([
            DatasetItem(1, annotations=[ Label(0) ])
        ], categories=['a'])

        src3 = Dataset.from_iterable([
            DatasetItem(1, annotations=[ Caption('hello world') ])
        ])

        actual = Dataset.from_extractors(src1, src2, src3)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'])

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            loaded_dataset = Dataset.load(test_dir)

            compare_datasets(self, source_dataset, loaded_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        env = Environment()
        env.importers.items = {DEFAULT_FORMAT: env.importers[DEFAULT_FORMAT]}
        env.extractors.items = {DEFAULT_FORMAT: env.extractors[DEFAULT_FORMAT]}

        dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'])

        with TestDir() as test_dir:
            dataset.save(test_dir)

            detected_format = Dataset.detect(test_dir, env=env)

            self.assertEqual(DEFAULT_FORMAT, detected_format)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_and_import(self):
        env = Environment()
        env.importers.items = {DEFAULT_FORMAT: env.importers[DEFAULT_FORMAT]}
        env.extractors.items = {DEFAULT_FORMAT: env.extractors[DEFAULT_FORMAT]}

        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'])

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            imported_dataset = Dataset.import_from(test_dir, env=env)

            self.assertEqual(imported_dataset.data_path, test_dir)
            self.assertEqual(imported_dataset.format, DEFAULT_FORMAT)
            compare_datasets(self, source_dataset, imported_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_by_string_format_name(self):
        env = Environment()
        env.converters.items = {'qq': env.converters[DEFAULT_FORMAT]}

        dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'], env=env)

        with TestDir() as test_dir:
            dataset.export(format='qq', save_dir=test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_scratch(self):
        dataset = Dataset()

        dataset.put(DatasetItem(1))
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3))
        dataset.remove(1)

        self.assertEqual(2, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_extractor(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                ]

        dataset = Dataset.from_extractors(TestExtractor())

        self.assertEqual(3, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_length_when_created_from_sequence(self):
        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
            DatasetItem(3),
        ])

        self.assertEqual(3, len(dataset))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform_by_string_name(self):
        expected = Dataset.from_iterable([
            DatasetItem(id=1, attributes={'qq': 1}),
        ])

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, attributes={'qq': 1})

        env = Environment()
        env.transforms.register('qq', TestTransform)

        dataset = Dataset.from_iterable([ DatasetItem(id=1) ], env=env)

        actual = dataset.transform('qq')

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform(self):
        expected = Dataset.from_iterable([
            DatasetItem(id=1, attributes={'qq': 1}),
        ])

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, attributes={'qq': 1})

        dataset = Dataset.from_iterable([ DatasetItem(id=1) ])

        actual = dataset.transform(TestTransform)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_annotations(self):
        a = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', annotations=[
                Label(1, id=3),
                Label(2, attributes={ 'x': 1 }),
            ])
        ], categories=['a', 'b', 'c', 'd'])

        b = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', annotations=[
                Label(2, attributes={ 'x': 1 }),
                Label(3, id=4),
            ])
        ], categories=['a', 'b', 'c', 'd'])

        expected = Dataset.from_iterable([
            DatasetItem(id=1, subset='train', annotations=[
                Label(1, id=3),
                Label(2, attributes={ 'x': 1 }),
                Label(3, id=4),
            ])
        ], categories=['a', 'b', 'c', 'd'])

        merged = Dataset.from_extractors(a, b)

        compare_datasets(self, expected, merged)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_join_different_categories(self):
        s1 = Dataset.from_iterable([], categories=['a', 'b'])
        s2 = Dataset.from_iterable([], categories=['b', 'a'])

        with self.assertRaisesRegex(DatumaroError, "different categories"):
            Dataset.from_extractors(s1, s2)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_join_datasets(self):
        s1 = Dataset.from_iterable([ DatasetItem(0), DatasetItem(1) ])
        s2 = Dataset.from_iterable([ DatasetItem(1), DatasetItem(2) ])
        expected = Dataset.from_iterable([
            DatasetItem(0), DatasetItem(1), DatasetItem(2)
        ])

        actual = Dataset.from_extractors(s1, s2)

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable([
                DatasetItem(1, subset='a'),
                DatasetItem(2, subset='b'),
                DatasetItem(3, subset='c'),
            ])
            dataset.save(path)
            os.unlink(osp.join(
                path, 'annotations', 'a.json')) # should be rewritten
            os.unlink(osp.join(
                path, 'annotations', 'b.json')) # should not be rewritten
            os.unlink(osp.join(
                path, 'annotations', 'c.json')) # should not be rewritten

            dataset.put(DatasetItem(2, subset='a'))
            dataset.remove(3, 'c')
            dataset.save()

            self.assertTrue(osp.isfile(osp.join(path, 'annotations', 'a.json')))
            self.assertFalse(osp.isfile(osp.join(path, 'annotations', 'b.json')))
            self.assertTrue(osp.isfile(osp.join(path, 'annotations', 'c.json')))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_track_modifications_on_addition(self):
        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
        ])

        self.assertFalse(dataset.is_modified)

        dataset.put(DatasetItem(3, subset='a'))

        self.assertTrue(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_track_modifications_on_removal(self):
        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
        ])

        self.assertFalse(dataset.is_modified)

        dataset.remove(1)

        self.assertTrue(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_patch(self):
        expected = Dataset.from_iterable([
            DatasetItem(2),
            DatasetItem(3, subset='a')
        ])

        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
        ])
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3, subset='a'))
        dataset.remove(1)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.added,
            ('3', 'a'): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_create_patch_when_cached(self):
        expected = Dataset.from_iterable([
            DatasetItem(2),
            DatasetItem(3, subset='a')
        ])

        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
        ])
        dataset.init_cache()
        dataset.put(DatasetItem(2))
        dataset.put(DatasetItem(3, subset='a'))
        dataset.remove(1)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,

            # Item was not changed from the original one.
            # TODO: add item comparison and remove this line
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.modified,

            ('3', 'a'): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_mixed(self):
        expected = Dataset.from_iterable([
            DatasetItem(2),
            DatasetItem(3, subset='a')
        ])

        dataset = Dataset.from_iterable([
            DatasetItem(1),
            DatasetItem(2),
        ])

        class Remove1(Transform):
            def __iter__(self):
                for item in self._extractor:
                    if item.id != '1':
                        yield item

        class Add3(Transform):
            def __iter__(self):
                for item in self._extractor:
                    if item.id == '2':
                        yield item
                yield DatasetItem(3, subset='a')

        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.modified,
            ('3', 'a'): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_chained(self):
        expected = Dataset.from_iterable([
            DatasetItem(2),
            DatasetItem(3, subset='a')
        ])

        class TestExtractor(Extractor):
            iter_called = 0
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0
            def __iter__(self):
                for item in self._extractor:
                    if item.id != '1':
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0
            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset='a')

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.modified,
            ('3', 'a'): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))

        self.assertEqual(TestExtractor.iter_called, 2) # 1 for items, 1 for list
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_intermixed_with_direct_ops(self):
        expected = Dataset.from_iterable([
            DatasetItem(3, subset='a'),
            DatasetItem(4),
            DatasetItem(5),
        ])

        class TestExtractor(Extractor):
            iter_called = 0
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0
            def __iter__(self):
                for item in self._extractor:
                    if item.id != '1':
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0
            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset='a')

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.init_cache()
        dataset.put(DatasetItem(4))
        dataset.transform(Remove1)
        dataset.put(DatasetItem(5))
        dataset.remove(2)
        dataset.transform(Add3)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('3', 'a'): ItemStatus.added,
            ('4', DEFAULT_SUBSET_NAME): ItemStatus.added,
            ('5', DEFAULT_SUBSET_NAME): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(3, len(patch.data))

        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(None, patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))
        self.assertEqual(dataset.get(4), patch.data.get(4))
        self.assertEqual(dataset.get(5), patch.data.get(5))

        self.assertEqual(TestExtractor.iter_called, 1)
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_local_transforms_stacked(self):
        expected = Dataset.from_iterable([
            DatasetItem(4),
            DatasetItem(5),
        ])

        class TestExtractor(Extractor):
            iter_called = 0
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class ShiftIds(ItemTransform):
            def transform_item(self, item):
                return item.wrap(id=int(item.id) + 1)

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.remove(2)
        dataset.transform(ShiftIds)
        dataset.transform(ShiftIds)
        dataset.transform(ShiftIds)
        dataset.put(DatasetItem(5))

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('4', DEFAULT_SUBSET_NAME): ItemStatus.added,
            ('5', DEFAULT_SUBSET_NAME): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))

        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(None, patch.data.get(2))
        self.assertEqual(None, patch.data.get(3))
        self.assertEqual(dataset.get(4), patch.data.get(4))
        self.assertEqual(dataset.get(5), patch.data.get(5))

        self.assertEqual(TestExtractor.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_can_create_patch_when_transforms_chained_and_source_cached(self):
        expected = Dataset.from_iterable([
            DatasetItem(2),
            DatasetItem(3, subset='a')
        ])

        class TestExtractor(Extractor):
            iter_called = 0
            def __iter__(self):
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                ]

                __class__.iter_called += 1

        class Remove1(Transform):
            iter_called = 0
            def __iter__(self):
                for item in self._extractor:
                    if item.id != '1':
                        yield item

                __class__.iter_called += 1

        class Add3(Transform):
            iter_called = 0
            def __iter__(self):
                yield from self._extractor
                yield DatasetItem(3, subset='a')

                __class__.iter_called += 1

        dataset = Dataset.from_extractors(TestExtractor())
        dataset.init_cache()
        dataset.transform(Remove1)
        dataset.transform(Add3)

        patch = dataset.patch

        self.assertEqual({
            ('1', DEFAULT_SUBSET_NAME): ItemStatus.removed,
            ('2', DEFAULT_SUBSET_NAME): ItemStatus.modified, # TODO: remove this
            ('3', 'a'): ItemStatus.added,
        }, patch.updated_items)

        self.assertEqual({
            'default': ItemStatus.modified,
            'a': ItemStatus.modified,
        }, patch.updated_subsets)

        self.assertEqual(2, len(patch.data))
        self.assertEqual(None, patch.data.get(1))
        self.assertEqual(dataset.get(2), patch.data.get(2))
        self.assertEqual(dataset.get(3, 'a'), patch.data.get(3, 'a'))

        self.assertEqual(TestExtractor.iter_called, 1) # 1 for items and list
        self.assertEqual(Remove1.iter_called, 1)
        self.assertEqual(Add3.iter_called, 1)

        compare_datasets(self, expected, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_put_and_remove(self):
        iter_called = False
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter([
                    DatasetItem(1),
                    DatasetItem(2),
                ])
        dataset = Dataset.from_extractors(TestExtractor())

        self.assertFalse(dataset.is_cache_initialized)

        dataset.put(DatasetItem(3))
        dataset.remove(DatasetItem(1))

        self.assertFalse(dataset.is_cache_initialized)
        self.assertFalse(iter_called)

        dataset.init_cache()

        self.assertTrue(dataset.is_cache_initialized)
        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_put(self):
        dataset = Dataset()

        dataset.put(DatasetItem(1))

        self.assertTrue((1, '') in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_get_on_updated_item(self):
        iter_called = False
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter([
                    DatasetItem(1),
                    DatasetItem(2),
                ])
        dataset = Dataset.from_extractors(TestExtractor())

        dataset.put(DatasetItem(2))

        self.assertTrue((2, '') in dataset)
        self.assertFalse(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_switch_eager_and_lazy_with_cm_global(self):
        iter_called = False
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                return iter([
                    DatasetItem(1),
                    DatasetItem(2),
                ])

        with eager_mode():
            Dataset.from_extractors(TestExtractor())

        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_switch_eager_and_lazy_with_cm_local(self):
        iter_called = False
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called = True
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]
        dataset = Dataset.from_extractors(TestExtractor())

        with eager_mode(dataset=dataset):
            dataset.select(lambda item: int(item.id) < 3)
            dataset.select(lambda item: int(item.id) < 2)

        self.assertTrue(iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_lazy_select(self):
        iter_called = 0
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]
        dataset = Dataset.from_extractors(TestExtractor())

        dataset.select(lambda item: int(item.id) < 3)
        dataset.select(lambda item: int(item.id) < 2)

        self.assertEqual(iter_called, 0)

        self.assertEqual(1, len(dataset))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_chain_lazy_transforms(self):
        iter_called = 0
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                yield from [
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ]
        dataset = Dataset.from_extractors(TestExtractor())

        class TestTransform(ItemTransform):
            def transform_item(self, item):
                return self.wrap_item(item, id=int(item.id) + 1)

        dataset.transform(TestTransform)
        dataset.transform(TestTransform)

        self.assertEqual(iter_called, 0)

        self.assertEqual(4, len(dataset))
        self.assertEqual(3, int(min(int(item.id) for item in dataset)))

        self.assertEqual(iter_called, 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_raises_when_repeated_items_in_source(self):
        dataset = Dataset.from_iterable([DatasetItem(0), DatasetItem(0)])

        with self.assertRaises(RepeatedItemError):
            dataset.init_cache()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_check_item_existence(self):
        dataset = Dataset.from_iterable([
            DatasetItem(0, subset='a'), DatasetItem(1)
        ])

        self.assertTrue(DatasetItem(0, subset='a') in dataset)
        self.assertFalse(DatasetItem(0, subset='b') in dataset)
        self.assertTrue((0, 'a') in dataset)
        self.assertFalse((0, 'b') in dataset)
        self.assertTrue(1 in dataset)
        self.assertFalse(0 in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_put_with_id_override(self):
        dataset = Dataset.from_iterable([])

        dataset.put(DatasetItem(0, subset='a'), id=2, subset='b')

        self.assertTrue((2, 'b') in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_compute_cache_with_empty_source(self):
        dataset = Dataset.from_iterable([])
        dataset.put(DatasetItem(2))

        dataset.init_cache()

        self.assertTrue(2 in dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_do_partial_caching_in_get_when_default(self):
        iter_called = 0
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                return iter([
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ])

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.get(3)
        dataset.get(4)

        self.assertEqual(1, iter_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_do_partial_caching_in_get_when_redefined(self):
        iter_called = 0
        get_called = 0
        class TestExtractor(Extractor):
            def __iter__(self):
                nonlocal iter_called
                iter_called += 1
                return iter([
                    DatasetItem(1),
                    DatasetItem(2),
                    DatasetItem(3),
                    DatasetItem(4),
                ])

            def get(self, id, subset=None): #pylint: disable=redefined-builtin
                nonlocal get_called
                get_called += 1
                return DatasetItem(id, subset=subset)

        dataset = Dataset.from_extractors(TestExtractor())

        dataset.get(3)
        dataset.get(4)

        self.assertEqual(0, iter_called)
        self.assertEqual(2, get_called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_binds_on_save(self):
        dataset = Dataset.from_iterable([DatasetItem(1)])

        self.assertFalse(dataset.is_bound)

        with TestDir() as test_dir:
            dataset.save(test_dir)

            self.assertTrue(dataset.is_bound)
            self.assertEqual(dataset.data_path, test_dir)
            self.assertEqual(dataset.format, DEFAULT_FORMAT)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_flushes_changes_on_save(self):
        dataset = Dataset.from_iterable([])
        dataset.put(DatasetItem(1))

        self.assertTrue(dataset.is_modified)

        with TestDir() as test_dir:
            dataset.save(test_dir)

            self.assertFalse(dataset.is_modified)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_does_not_load_images_on_saving(self):
        # Issue https://github.com/openvinotoolkit/datumaro/issues/177
        # Missing image metadata (size etc.) can lead to image loading on
        # dataset save without image saving

        called = False
        def test_loader():
            nonlocal called
            called = True

        dataset = Dataset.from_iterable([
            DatasetItem(1, image=test_loader)
        ])

        with TestDir() as test_dir:
            dataset.save(test_dir)

        self.assertFalse(called)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_transform_labels(self):
        expected = Dataset.from_iterable([], categories=['c', 'b'])
        dataset = Dataset.from_iterable([], categories=['a', 'b'])

        actual = dataset.transform('remap_labels', {'a': 'c'})

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_run_model(self):
        dataset = Dataset.from_iterable([
            DatasetItem(i, image=np.array([i]))
            for i in range(5)
        ], categories=['label'])

        batch_size = 3

        expected = Dataset.from_iterable([
            DatasetItem(i, image=np.array([i]), annotations=[
                Label(0, attributes={ 'idx': i % batch_size, 'data': i })
            ])
            for i in range(5)
        ], categories=['label'])

        calls = 0

        class TestLauncher(Launcher):
            def launch(self, inputs):
                nonlocal calls
                calls += 1

                for i, inp in enumerate(inputs):
                    yield [ Label(0, attributes={'idx': i, 'data': inp.item()}) ]

        model = TestLauncher()

        actual = dataset.run_model(model, batch_size=batch_size)

        compare_datasets(self, expected, actual, require_images=True)
        self.assertEqual(2, calls)

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_items(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='train'),
            DatasetItem(id=1, subset='test'),
        ])

        dataset.filter('/item[id > 0]')

        self.assertEqual(1, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_257)
    def test_filter_registers_changes(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='train'),
            DatasetItem(id=1, subset='test'),
        ])

        dataset.filter('/item[id > 0]')

        self.assertEqual({
            ('0', 'train'): ItemStatus.removed,
            ('1', 'test'): ItemStatus.modified, # TODO: remove this line
        }, dataset.patch.updated_items)

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_annotations(self):
        dataset = Dataset.from_iterable([
            DatasetItem(id=0, subset='train', annotations=[Label(0), Label(1)]),
            DatasetItem(id=1, subset='val', annotations=[Label(2)]),
            DatasetItem(id=2, subset='test', annotations=[Label(0), Label(2)]),
        ], categories=['a', 'b', 'c'])

        dataset.filter('/item/annotation[label = "c"]',
            filter_annotations=True, remove_empty=True)

        self.assertEqual(2, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_items_in_merged_dataset(self):
        dataset = Dataset.from_extractors(
            Dataset.from_iterable([ DatasetItem(id=0, subset='train') ]),
            Dataset.from_iterable([ DatasetItem(id=1, subset='test') ]),
        )

        dataset.filter('/item[id > 0]')

        self.assertEqual(1, len(dataset))

    @mark_requirement(Requirements.DATUM_BUG_259)
    def test_can_filter_annotations_in_merged_dataset(self):
        dataset = Dataset.from_extractors(
            Dataset.from_iterable([
                DatasetItem(id=0, subset='train', annotations=[Label(0)]),
            ], categories=['a', 'b', 'c']),
            Dataset.from_iterable([
                DatasetItem(id=1, subset='val', annotations=[Label(1)]),
            ], categories=['a', 'b', 'c']),
            Dataset.from_iterable([
                DatasetItem(id=2, subset='test', annotations=[Label(2)]),
            ], categories=['a', 'b', 'c']),
        )

        dataset.filter('/item/annotation[label = "c"]',
            filter_annotations=True, remove_empty=True)

        self.assertEqual(1, len(dataset))


class DatasetItemTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctor_requires_id(self):
        with self.assertRaises(Exception):
            # pylint: disable=no-value-for-parameter
            DatasetItem()
            # pylint: enable=no-value-for-parameter

    @staticmethod
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctors_with_image():
        for args in [
            { 'id': 0, 'image': None },
            { 'id': 0, 'image': 'path.jpg' },
            { 'id': 0, 'image': np.array([1, 2, 3]) },
            { 'id': 0, 'image': lambda f: np.array([1, 2, 3]) },
            { 'id': 0, 'image': Image(data=np.array([1, 2, 3])) },
        ]:
            DatasetItem(**args)


class DatasetFilterTest(TestCase):
    @staticmethod
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_item_representations():
        item = DatasetItem(id=1, subset='subset', path=['a', 'b'],
            image=np.ones((5, 4, 3)),
            annotations=[
                Label(0, attributes={'a1': 1, 'a2': '2'}, id=1, group=2),
                Caption('hello', id=1),
                Caption('world', group=5),
                Label(2, id=3, attributes={ 'x': 1, 'y': '2' }),
                Bbox(1, 2, 3, 4, label=4, id=4, attributes={ 'a': 1.0 }),
                Bbox(5, 6, 7, 8, id=5, group=5),
                Points([1, 2, 2, 0, 1, 1], label=0, id=5),
                Mask(id=5, image=np.ones((3, 2))),
                Mask(label=3, id=5, image=np.ones((2, 3))),
                PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11),
                Polygon([1, 2, 3, 4, 5, 6, 7, 8]),
            ]
        )

        encoded = DatasetItemEncoder.encode(item)
        DatasetItemEncoder.to_string(encoded)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_item_filter_can_be_applied(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                for i in range(4):
                    yield DatasetItem(id=i, subset='train')

        extractor = TestExtractor()

        filtered = XPathDatasetFilter(extractor, '/item[id > 1]')

        self.assertEqual(2, len(filtered))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotations_filter_can_be_applied(self):
        class SrcExtractor(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=0),
                    DatasetItem(id=1, annotations=[
                        Label(0),
                        Label(1),
                    ]),
                    DatasetItem(id=2, annotations=[
                        Label(0),
                        Label(2),
                    ]),
                ])

        class DstExtractor(Extractor):
            def __iter__(self):
                return iter([
                    DatasetItem(id=0),
                    DatasetItem(id=1, annotations=[
                        Label(0),
                    ]),
                    DatasetItem(id=2, annotations=[
                        Label(0),
                    ]),
                ])

        extractor = SrcExtractor()

        filtered = XPathAnnotationsFilter(extractor,
            '/item/annotation[label_id = 0]')

        self.assertListEqual(list(filtered), list(DstExtractor()))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_annotations_filter_can_remove_empty_items(self):
        source = Dataset.from_iterable([
            DatasetItem(id=0),
            DatasetItem(id=1, annotations=[
                Label(0),
                Label(1),
            ]),
            DatasetItem(id=2, annotations=[
                Label(0),
                Label(2),
            ]),
        ], categories=['a', 'b', 'c'])

        expected = Dataset.from_iterable([
            DatasetItem(id=2, annotations=[Label(2)]),
        ], categories=['a', 'b', 'c'])

        filtered = XPathAnnotationsFilter(source,
            '/item/annotation[label_id = 2]', remove_empty=True)

        compare_datasets(self, expected, filtered)
