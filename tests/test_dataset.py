import numpy as np

from unittest import TestCase

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import (Extractor, DatasetItem,
    Label, Mask, Points, Polygon, PolyLine, Bbox, Caption,
)
from datumaro.util.image import Image
from datumaro.components.dataset_filter import \
    XPathDatasetFilter, XPathAnnotationsFilter, DatasetItemEncoder
from datumaro.util.test_utils import compare_datasets


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

    def test_can_join_annotations(self):
        class TestExtractor1(Extractor):
            def __iter__(self):
                yield DatasetItem(id=1, subset='train', annotations=[
                    Label(2, id=3),
                    Label(3, attributes={ 'x': 1 }),
                ])

        class TestExtractor2(Extractor):
            def __iter__(self):
                yield DatasetItem(id=1, subset='train', annotations=[
                    Label(3, attributes={ 'x': 1 }),
                    Label(4, id=4),
                ])

        merged = Dataset.from_extractors(TestExtractor1(), TestExtractor2())

        self.assertEqual(1, len(merged))

        item = next(iter(merged))
        self.assertEqual(3, len(item.annotations))

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
