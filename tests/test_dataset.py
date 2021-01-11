import numpy as np

from unittest import TestCase

from datumaro.components.environment import Environment
from datumaro.components.extractor import (Extractor, DatasetItem,
    Label, Mask, Points, Polygon, PolyLine, Bbox, Caption,
    LabelCategories, AnnotationType, Transform
)
from datumaro.util.image import Image
from datumaro.components.dataset_filter import \
    XPathDatasetFilter, XPathAnnotationsFilter, DatasetItemEncoder
from datumaro.components.dataset import Dataset, DEFAULT_FORMAT
from datumaro.util.test_utils import TestDir, compare_datasets


class DatasetTest(TestCase):
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

    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'])

        with TestDir() as test_dir:
            source_dataset.save(test_dir)

            loaded_dataset = Dataset.load(test_dir)

            compare_datasets(self, source_dataset, loaded_dataset)

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

            compare_datasets(self, source_dataset, imported_dataset)

    def test_can_export_by_string_format_name(self):
        env = Environment()
        env.converters.items = {'qq': env.converters[DEFAULT_FORMAT]}

        dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'], env=env)

        with TestDir() as test_dir:
            dataset.export('qq', save_dir=test_dir)

    def test_can_transform_by_string_name(self):
        expected = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ], attributes={'qq': 1}),
        ], categories=['a', 'b', 'c'])

        class TestTransform(Transform):
            def transform_item(self, item):
                return self.wrap_item(item, attributes={'qq': 1})

        env = Environment()
        env.transforms.items = {'qq': TestTransform}

        dataset = Dataset.from_iterable([
            DatasetItem(id=1, annotations=[ Label(2) ]),
        ], categories=['a', 'b', 'c'], env=env)

        actual = dataset.transform('qq')

        compare_datasets(self, expected, actual)

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

    def test_cant_join_different_categories(self):
        s1 = Dataset.from_iterable([], categories=['a', 'b'])
        s2 = Dataset.from_iterable([], categories=['b', 'a'])

        with self.assertRaisesRegex(Exception, "different categories"):
            Dataset.from_extractors(s1, s2)

    def test_can_join_datasets(self):
        s1 = Dataset.from_iterable([ DatasetItem(0), DatasetItem(1) ])
        s2 = Dataset.from_iterable([ DatasetItem(1), DatasetItem(2) ])

        dataset = Dataset.from_extractors(s1, s2)

        self.assertEqual(3, len(dataset))


class DatasetItemTest(TestCase):
    def test_ctor_requires_id(self):
        with self.assertRaises(Exception):
            # pylint: disable=no-value-for-parameter
            DatasetItem()
            # pylint: enable=no-value-for-parameter

    @staticmethod
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

    def test_item_filter_can_be_applied(self):
        class TestExtractor(Extractor):
            def __iter__(self):
                for i in range(4):
                    yield DatasetItem(id=i, subset='train')

        extractor = TestExtractor()

        filtered = XPathDatasetFilter(extractor, '/item[id > 1]')

        self.assertEqual(2, len(filtered))

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
