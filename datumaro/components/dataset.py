# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Iterable, Iterator, Optional, Set, Tuple, Union, Dict, List
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.extractor import (Categories, Extractor, IExtractor, LabelCategories,
    AnnotationType, DatasetItem, DEFAULT_SUBSET_NAME, Transform)
from datumaro.components.dataset_filter import \
    XPathDatasetFilter, XPathAnnotationsFilter
from datumaro.components.environment import Environment
from datumaro.util import error_rollback
from datumaro.util.log_utils import logging_disabled


DEFAULT_FORMAT = 'datumaro'

IDataset = IExtractor

class DatasetItemStorage:
    pass # forward declaration for type annotations

class DatasetItemStorage:
    def __init__(self):
        self.data = {} # { subset_name: { id: DatasetItem } }
        self._length = 0 # needed because removed items are masked

    def __iter__(self) -> Iterator[DatasetItem]:
        for subset in self.data.values():
            for item in subset.values():
                if item:
                    yield item

    def __len__(self) -> int:
        return self._length

    def put(self, item) -> bool:
        subset = self.data.setdefault(item.subset, {})
        is_new = subset.get(item.id) == None
        if is_new:
            self._length += 1
        subset[item.id] = item
        return is_new

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        subset = subset or DEFAULT_SUBSET_NAME
        return self.data.get(subset, {}).get(id)

    def remove(self, id, subset=None) -> bool:
        subset = self.data.get(subset, {})
        is_removed = subset.get(id) != None
        if is_removed:
            self._length -= 1
            subset[id] = None # mark removed
        return is_removed

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        return self.get(*x) is not None

    def is_removed(self, id, subset=None) -> bool:
        subset = subset or DEFAULT_SUBSET_NAME
        return self.data.get(subset, {}).get(id, object()) is None

    def get_subset(self, name):
        return self.data[name]

    def subsets(self):
        return self.data

class DatasetItemStorageDatasetView(IDataset):
    class Subset(IDataset):
        def __init__(self, parent, name):
            super().__init__()
            self.parent = parent
            self.name = name

        @property
        def _data(self):
            return self.parent._get_subset_data(self.name)

        def __iter__(self):
            yield from self._data.values()

        def __len__(self):
            return len(self._data)

        def put(self, item):
            return self._data.put(item)

        def get(self, id, subset=None):
            return self._data.get(id, subset)

        def remove(self, id, subset=None):
            return self._data.remove(id, subset)

        def get_subset(self, name):
            return __class__(self.parent, self._data.get_subset(name))

        def subsets(self):
            return { k: self.get_subset(k) for k in self._data.subsets() }

        def categories(self):
            return self.parent.categories()


    def __init__(self, parent: DatasetItemStorage, categories: Dict):
        self._parent = parent
        self._categories = categories

    def __iter__(self):
        yield from self._parent

    def __len__(self):
        return len(self._parent)

    def categories(self):
        return self._categories

    def get_subset(self, name):
        return self.Subset(self, name)

    def _get_subset_data(self, name):
        return self._parent.get_subset(name)

    def subsets(self):
        return {k: self.get_subset(k) for k in self._parent.subsets()}

    def get(self, id, subset=None):
        return self._parent.get(id, subset=subset)


class ExactMerge:
    @classmethod
    def merge(cls, *sources):
        items = DatasetItemStorage()
        for source in sources:
            for item in source:
                existing_item = items.get(item.id, item.subset)
                if existing_item is not None:
                    path = existing_item.path
                    if item.path != path:
                        path = None
                    item = cls.merge_items(existing_item, item, path=path)

                items.put(item)
        return items

    @staticmethod
    def _lazy_image(item):
        # NOTE: avoid https://docs.python.org/3/faq/programming.html#why-do-lambdas-defined-in-a-loop-with-different-values-all-return-the-same-result
        return lambda: item.image

    @classmethod
    def merge_items(cls, existing_item, current_item, path=None):
        return existing_item.wrap(path=path,
            image=cls.merge_images(existing_item, current_item),
            annotations=cls.merge_anno(
                existing_item.annotations, current_item.annotations))

    @staticmethod
    def merge_images(existing_item, current_item):
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
                assert existing_item.image._size == current_item.image._size, \
                    "Image size info differs for item '%s': %s vs %s" % \
                    (existing_item.id,
                     existing_item.image._size, current_item.image._size)
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
    def merge_anno(a, b):
        from .operations import merge_annotations_equal
        return merge_annotations_equal(a, b)

    @staticmethod
    def merge_categories(sources):
        from .operations import merge_categories
        return merge_categories(sources)


class Dataset(IDataset):
    pass # forward declaration for type annotations

class DatasetSubset(IDataset): # non-owning view
    def __init__(self, parent: Dataset, name: str):
        super().__init__()
        self.parent = parent
        self.name = name

    def __iter__(self):
        yield from self.parent._data.get_subset(self.name)

    def __len__(self):
        return len(self.parent._data.get_subset(self.name))

    def put(self, item):
        return self.parent.put(item, subset=self.name)

    def get(self, id, subset=None):
        assert subset is None
        return self.parent.get(id, subset=self.name)

    def remove(self, id, subset=None):
        assert subset is None
        return self.parent.remove(id, subset=self.name)

    def get_subset(self, name):
        assert not name
        return self

    def subsets(self):
        return {}

    def categories(self):
        return self.parent.categories()


class DatasetStorage(IDataset):
    pass # forward declaration for type annotations

class DatasetStorage(IDataset):
    _UPDATED_ALL = 'all'

    def __init__(self, source: IDataset = None, categories=None):
        if source is None and categories is None:
            categories = {}
        self._categories = categories

        # possible combinations
        # 1. source + storage (patch)
        # 2. no source + storage
        #      cache or just a dataset from scratch, or cached transform
        #  - In this case updated_items describes the patch
        self._source = source
        self._storage = DatasetItemStorage() # patch or cache
        self._updated_items = set() # set or UPDATED_ALL

        self._length = None

    def is_cache_initialized(self) -> bool:
        return self._source is None

    @property
    def _is_unchanged_wrapper(self) -> bool:
        return self._source is not None and not self._updated_items

    def init_cache(self):
        if self._is_unchanged_wrapper:
            for _ in self._iter_init_cache(): pass
        elif self._source:
            merged = ExactMerge.merge(self._source, self._storage)
            for item in merged:
                if self._storage.is_removed(item.id, item.subset):
                    merged.remove(item.id, item.subset)
            self._storage = merged
            if self._categories is None:
                self._categories = self._source.categories()
            self._source = None

        self._length = len(self._storage)

    def _iter_init_cache(self) -> Iterable[DatasetItem]:
        # If iterated in parallel, the result is undefined.
        # If storage is changed during iteration, the result is undefined.
        #
        # TODO: can potentially be optimized by sharing
        # the cache between parallel consumers and introducing some kind of lock

        cache = DatasetItemStorage()

        for item in self._source:
            if item in cache:
                raise Exception(
                    "Item (%s, %s) repeats in the source dataset" % \
                    (item.id, item.subset)
                )
            cache.put(item)
            yield item

        if len(self._storage) != 0:
            return
        self._storage = cache
        if self._categories is None:
            self._categories = self._source.categories()
        self._length = len(cache)
        self._source = None

    def __iter__(self) -> Iterable[DatasetItem]:
        if self._is_unchanged_wrapper:
            yield from self._iter_init_cache()
        else:
            yield from self._merged()

    def _merged(self) -> IDataset:
        if self._is_unchanged_wrapper:
            return self._source
        elif self._source:
            self.init_cache()
        return DatasetItemStorageDatasetView(self._storage, self._categories)

    def __len__(self) -> int:
        if self._length is None:
            self.init_cache()
        return self._length

    def categories(self) -> Dict[AnnotationType, Categories]:
        if self._categories is not None:
            return self._categories
        else:
            return self._source.categories()

    def put(self, item):
        is_new = self._storage.put(item)
        if is_new and not self.is_cache_initialized():
            self._length = None
        if self._length is not None:
            self._length += is_new

    def get(self, id, subset) -> Optional[DatasetItem]:
        item = self._storage.get(id, subset)
        if item is None and not self.is_cache_initialized():
            if getattr(self._source, 'get') == Extractor.get:
                # can be improved if IDataset is ABC
                self.init_cache()
                item = self._storage.get(id, subset)
            else:
                item = self._source.get(id, subset)
                self._storage.put(item)
        return item

    def remove(self, id, subset=None):
        is_removed = self._storage.remove(id, subset)
        if is_removed and not self.is_cache_initialized():
            self._length = None
        if self._length is not None:
            self._length -= is_removed

    def get_subset(self, name):
        return self._merged().get_subset(name)

    def subsets(self):
        subsets = {}
        if not self.is_cache_initialized():
            subsets.update(self._source.subsets())
        subsets.update(self._storage.subsets())
        return subsets

    def transform(self, method, **kwargs):
        self._source = method(self._merged(), **kwargs)
        self._storage = DatasetItemStorage()
        self._updated_items = self._UPDATED_ALL

    def get_patch(self):
        if self._updated_items == self._UPDATED_ALL:
            self.init_cache()
            self._updated_items = set((item.id, item.subset)
                for item in self._storage)
        return self._storage


class Dataset(IDataset):
    @classmethod
    def from_iterable(cls, iterable: Iterable[DatasetItem],
            categories: Union[Dict, List[str]] = None,
            env: Environment = None):
        if isinstance(categories, list):
            categories = { AnnotationType.label:
                LabelCategories.from_iterable(categories)
            }

        if not categories:
            categories = {}

        class _extractor(Extractor):
            def __init__(self):
                super().__init__(length=len(iterable) \
                    if hasattr(iterable, '__len__') else None)

            def __iter__(self):
                return iter(iterable)

            def categories(self):
                return categories

        return cls.from_extractors(_extractor(), env=env)

    @staticmethod
    def from_extractors(*sources: IDataset, env: Environment = None) -> Dataset:
        if len(sources) == 1:
            source = sources[0]
            categories = None
        else:
            source = ExactMerge.merge(*sources)
            categories = ExactMerge.merge_categories(
                s.categories() for s in sources)

        return Dataset(source=source, categories=categories, env=env)

    def __init__(self, source: IDataset = None, categories: Dict = None,
            env: Environment = None):
        super().__init__()

        assert env is None or isinstance(env, Environment), env
        self._env = env

        self._data = DatasetStorage(source, categories=categories)

        self._format = DEFAULT_FORMAT
        self._source_path = None

    def define_categories(self, categories: Dict):
        assert not self._data._categories and self._data._source is None
        self._data._categories = categories

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)

    def get_subset(self, name):
        return DatasetSubset(self, name)

    def subsets(self):
        return { k: self.get_subset(k) for k in self._data.subsets() }

    def categories(self):
        return self._data.categories()

    def get(self, id, subset=None):
        return self._data.get(id, subset)

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        return self.get(*x) is not None

    def put(self, item, id=None, subset=None):
        overrides = {}
        if id is not None:
            overrides['id'] = id
        if subset is not None:
            overrides['subset'] = subset
        if overrides:
            item = item.wrap(**overrides)

        self._data.put(item)

    def remove(self, id, subset=None):
        self._data.remove(id, subset)

    def filter(self, expr: str, filter_annotations: bool = False,
            remove_empty: bool = False) -> Dataset:
        if filter_annotations:
            return self.transform(XPathAnnotationsFilter, expr, remove_empty)
        else:
            return self.transform(XPathDatasetFilter, expr)

    def update(self, items: Iterable[DatasetItem]) -> Dataset:
        for item in items:
            self.put(item)
        return self

    def transform(self, method: Union[str, Transform], **kwargs) -> Dataset:
        if isinstance(method, str):
            method = self.env.make_transform(method)

        self._data.transform(method, **kwargs)
        return self

    def run_model(self, model, batch_size=1) -> Dataset:
        from datumaro.components.launcher import Launcher, ModelTransform
        if isinstance(model, Launcher):
            return self.transform(ModelTransform, launcher=model,
                batch_size=batch_size)
        elif isinstance(model, ModelTransform):
            return self.transform(model, batch_size=batch_size)
        else:
            raise TypeError('Unexpected model argument type: %s' % type(model))

    @property
    def data_path(self) -> Optional[str]:
        return self._source_path

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def updated_items(self) -> Set[DatasetItem]:
        return set((item.id, item.subset) for item in self._data.get_patch())

    @property
    def patch(self) -> IDataset:
        return DatasetItemStorageDatasetView(self._data.get_patch(),
            self._data._categories)

    @property
    def env(self) -> Environment:
        if not self._env:
            self._env = Environment()
        return self._env

    @error_rollback('on_error', implicit=True)
    def export(self, save_dir: str, format, **kwargs): #pylint: disable=redefined-builtin
        inplace = (save_dir == self._source_path and format == self._format)

        if isinstance(format, str):
            converter = self.env.converters[format]
        else:
            converter = format

        save_dir = osp.abspath(save_dir)
        if not osp.exists(save_dir):
            on_error.do(shutil.rmtree, save_dir, ignore_errors=True)
            inplace = False
        os.makedirs(save_dir, exist_ok=True)

        if not inplace:
            converter.convert(self, save_dir=save_dir, **kwargs)
        else:
            converter.patch(self, self.patch, save_dir=save_dir, **kwargs)

    def save(self, save_dir: str = None, **kwargs):
        self.export(save_dir or self._source_path,
            format=self._format, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> Dataset:
        return cls.import_from(path, format=DEFAULT_FORMAT, **kwargs)

    @classmethod
    def import_from(cls, path: str, format: str = None, \
            env: Environment = None, \
            **kwargs) -> Dataset: #pylint: disable=redefined-builtin
        from datumaro.components.config_model import Source

        if env is None:
            env = Environment()

        if not format:
            format = cls.detect(path, env)

        # TODO: remove importers, put this logic into extractors
        if format in env.importers:
            importer = env.make_importer(format)
            with logging_disabled(log.INFO):
                project = importer(path, **kwargs)
            detected_sources = list(project.config.sources.values())
        elif format in env.extractors:
            detected_sources = [{
                'url': path, 'format': format, 'options': kwargs
            }]
        else:
            raise Exception("Unknown source format '%s'. To make it "
                "available, add the corresponding Extractor implementation "
                "to the environment" % format)

        extractors = []
        for src_conf in detected_sources:
            if not isinstance(src_conf, Source):
                src_conf = Source(src_conf)
            extractors.append(env.make_extractor(
                src_conf.format, src_conf.url, **src_conf.options
            ))

        dataset = cls.from_extractors(*extractors, env=env)
        dataset._source_path = path
        dataset._format = format
        return dataset

    @staticmethod
    def detect(path: str, env: Environment = None) -> str:
        if env is None:
            env = Environment()

        matches = env.detect_dataset(path)
        if not matches:
            raise Exception("Failed to detect dataset format automatically: "
                "no matching formats found")
        if 1 < len(matches):
            raise Exception("Failed to detect dataset format automatically:"
                " data matches more than one format: %s" % \
                ', '.join(matches))
        return matches[0]
