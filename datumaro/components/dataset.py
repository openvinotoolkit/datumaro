# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict, defaultdict
from typing import Iterable, Union, Dict, List

from datumaro.components.extractor import (Extractor, LabelCategories,
    AnnotationType, DatasetItem, DEFAULT_SUBSET_NAME)
from datumaro.components.dataset_filter import \
    XPathDatasetFilter, XPathAnnotationsFilter
from datumaro.components.environment import Environment


class Dataset(Extractor):
    class Subset(Extractor):
        def __init__(self, parent):
            self.parent = parent
            self.items = OrderedDict()

        def __iter__(self):
            yield from self.items.values()

        def __len__(self):
            return len(self.items)

        def categories(self):
            return self.parent.categories()

    @classmethod
    def from_iterable(cls, iterable: Iterable[DatasetItem],
            categories: Union[Dict, List[str]] = None):
        if isinstance(categories, list):
            categories = { AnnotationType.label:
                LabelCategories.from_iterable(categories)
            }

        if not categories:
            categories = {}

        class _extractor(Extractor):
            def __iter__(self):
                return iter(iterable)

            def categories(self):
                return categories

        return cls.from_extractors(_extractor())

    @classmethod
    def from_extractors(cls, *sources):
        categories = cls._merge_categories(s.categories() for s in sources)
        dataset = Dataset(categories=categories)

        # merge items
        subsets = defaultdict(lambda: cls.Subset(dataset))
        for source in sources:
            for item in source:
                existing_item = subsets[item.subset].items.get(item.id)
                if existing_item is not None:
                    path = existing_item.path
                    if item.path != path:
                        path = None
                    item = cls._merge_items(existing_item, item, path=path)

                subsets[item.subset].items[item.id] = item

        dataset._subsets = dict(subsets)
        return dataset

    def __init__(self, categories=None):
        super().__init__()

        self._env = None

        self._subsets = {}

        if not categories:
            categories = {}
        self._categories = categories

    def __iter__(self):
        for subset in self._subsets.values():
            for item in subset:
                yield item

    def __len__(self):
        if self._length is None:
            self._length = sum(len(s) for s in self._subsets.values())
        return self._length

    def get_subset(self, name):
        return self._subsets[name]

    def subsets(self):
        return self._subsets

    def categories(self):
        return self._categories

    def get(self, item_id, subset=None, path=None):
        if path:
            raise KeyError("Requested dataset item path is not found")
        item_id = str(item_id)
        subset = subset or DEFAULT_SUBSET_NAME
        subset = self._subsets[subset]
        return subset.items[item_id]

    def put(self, item, item_id=None, subset=None, path=None):
        if path:
            raise KeyError("Requested dataset item path is not found")

        if item_id is None:
            item_id = item.id
        if subset is None:
            subset = item.subset

        item = item.wrap(id=item_id, subset=subset, path=None)
        if subset not in self._subsets:
            self._subsets[subset] = self.Subset(self)
        self._subsets[subset].items[item_id] = item
        self._length = None

        return item

    def filter(self, expr, filter_annotations=False, remove_empty=False):
        if filter_annotations:
            return self.transform(XPathAnnotationsFilter, expr, remove_empty)
        else:
            return self.transform(XPathDatasetFilter, expr)

    def update(self, items):
        for item in items:
            self.put(item)
        return self

    def define_categories(self, categories):
        assert not self._categories
        self._categories = categories

    @staticmethod
    def _lazy_image(item):
        # NOTE: avoid https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        return lambda: item.image

    @classmethod
    def _merge_items(cls, existing_item, current_item, path=None):
        return existing_item.wrap(path=path,
            image=cls._merge_images(existing_item, current_item),
            annotations=cls._merge_anno(
                existing_item.annotations, current_item.annotations))

    @staticmethod
    def _merge_images(existing_item, current_item):
        image = None
        if existing_item.has_image and current_item.has_image:
            if existing_item.image.has_data:
                image = existing_item.image
            else:
                image = current_item.image

            if existing_item.image.path != current_item.image.path:
                if not existing_item.image.path:
                    image._path = current_item.image.path

            if all([existing_item.image._size, current_item.image._size]):
                assert existing_item.image._size == current_item.image._size, "Image info differs for item '%s'" % existing_item.id
            elif existing_item.image._size:
                image._size = existing_item.image._size
            else:
                image._size = current_item.image._size
        elif existing_item.has_image:
            image = existing_item.image
        else:
            image = current_item.image

        return image

    @staticmethod
    def _merge_anno(a, b):
        # TODO: implement properly with merging and annotations remapping
        from .operations import merge_annotations_equal
        return merge_annotations_equal(a, b)

    @staticmethod
    def _merge_categories(sources):
        # TODO: implement properly with merging and annotations remapping
        from .operations import merge_categories
        return merge_categories(sources)

    def export(self, converter, save_dir):
        if isinstance(converter, str):
            converter = self.env.make_converter(converter)

        save_dir = osp.abspath(save_dir)
        save_dir_existed = osp.exists(save_dir)
        try:
            os.makedirs(save_dir, exist_ok=True)
            converter(self, save_dir)
        except BaseException:
            if not save_dir_existed:
                shutil.rmtree(save_dir)
            raise

    def transform(self, method, *args, **kwargs):
        if isinstance(method, str):
            method = self.env.make_transform(method)

        return super().transform(method, **kwargs)

    @property
    def env(self):
        if not self._env:
            self._env = Environment()
        return self._env
