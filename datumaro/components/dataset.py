# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

#pylint: disable=redefined-builtin

from contextlib import contextmanager
from enum import Enum
from typing import Iterable, Iterator, Optional, Tuple, Union, Dict, List
import logging as log
import os
import os.path as osp
import shutil

from datumaro.components.dataset_filter import \
    XPathDatasetFilter, XPathAnnotationsFilter
from datumaro.components.extractor import (CategoriesInfo, Extractor,
    IExtractor, LabelCategories, AnnotationType, DatasetItem,
    DEFAULT_SUBSET_NAME, Transform)
from datumaro.components.environment import Environment
from datumaro.components.errors import DatumaroError, RepeatedItemError
from datumaro.util import error_rollback
from datumaro.util.log_utils import logging_disabled


DEFAULT_FORMAT = 'datumaro'

IDataset = IExtractor

class DatasetItemStorage:
    def __init__(self):
        self.data = {} # { subset_name: { id: DatasetItem } }
        self._traversal_order = {} # maintain the order of elements

    def __iter__(self) -> Iterator[DatasetItem]:
        for item in self._traversal_order.values():
            yield item

    def __len__(self) -> int:
        return len(self._traversal_order)

    def put(self, item) -> bool:
        subset = self.data.setdefault(item.subset, {})
        is_new = subset.get(item.id) == None
        self._traversal_order[(item.id, item.subset)] = item
        subset[item.id] = item
        return is_new

    def _get(self, id, subset=None, dummy=None):
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        return self.data.get(subset, {}).get(id, dummy)

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        return self._get(id, subset)

    def remove(self, id, subset=None) -> bool:
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        subset_data = self.data.get(subset, {})
        is_removed = subset_data.pop(id, None) is not None
        if is_removed:
            self._traversal_order.pop((id, subset))
        return is_removed

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        return self.get(*x) is not None

    def get_subset(self, name):
        return self.data.get(name, {})

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
            assert subset or DEFAULT_SUBSET_NAME == \
                   self.name or DEFAULT_SUBSET_NAME
            return self._data.get(id, subset)

        def remove(self, id, subset=None):
            assert subset or DEFAULT_SUBSET_NAME == \
                   self.name or DEFAULT_SUBSET_NAME
            return self._data.remove(id, subset)

        def get_subset(self, name):
            assert name or DEFAULT_SUBSET_NAME == \
                   self.name or DEFAULT_SUBSET_NAME
            return self

        def subsets(self):
            return { self.name or DEFAULT_SUBSET_NAME: self }

        def categories(self):
            return self.parent.categories()


    def __init__(self, parent: DatasetItemStorage, categories: CategoriesInfo):
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


ItemStatus = Enum('ItemStatus', ['added', 'modified', 'removed'])

class DatasetPatch:
    def __init__(self, data: DatasetItemStorage,
            categories: CategoriesInfo,
            updated_items: Dict[Tuple[str, str], ItemStatus],
            updated_subsets: Dict[str, ItemStatus] = None):
        self.data = data
        self.categories = categories
        self.updated_items = updated_items
        self._updated_subsets = updated_subsets

    @property
    def updated_subsets(self) -> Dict[str, ItemStatus]:
        if self._updated_subsets is None:
            subset_stats = set()
            for _, subset in self.updated_items:
                subset_stats.add(subset)
            self._updated_subsets = {
                subset: ItemStatus.modified for subset in subset_stats
            }
        return self._updated_subsets

    def as_dataset(self) -> IDataset:
        return DatasetItemStorageDatasetView(self.data, self.categories)


class DatasetSubset(IDataset): # non-owning view
    def __init__(self, parent: 'Dataset', name: str):
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
        assert subset or DEFAULT_SUBSET_NAME == \
               self.name or DEFAULT_SUBSET_NAME
        return self.parent.get(id, subset=self.name)

    def remove(self, id, subset=None):
        assert subset or DEFAULT_SUBSET_NAME == \
               self.name or DEFAULT_SUBSET_NAME
        return self.parent.remove(id, subset=self.name)

    def get_subset(self, name):
        assert name or DEFAULT_SUBSET_NAME == \
               self.name or DEFAULT_SUBSET_NAME
        return self

    def subsets(self):
        return { self.name or DEFAULT_SUBSET_NAME: self }

    def categories(self):
        return self.parent.categories()

    def as_dataset(self) -> 'Dataset':
        return Dataset.from_extractors(self, env=self.parent.env)


class DatasetStorage(IDataset):
    def __init__(self, source: IDataset = None,
            categories: CategoriesInfo = None):
        if source is None and categories is None:
            categories = {}
        elif source is not None and categories is not None:
            raise ValueError("Can't use both source and categories")
        self._categories = categories

        # possible combinations
        # 1. source + storage (patch)
        # 2. no source + storage
        #      cache or just a dataset from scratch, or cached transform
        #  - In this case updated_items describes the patch
        self._source = source
        self._storage = DatasetItemStorage() # patch or cache
        self._updated_items = {} # (id, subset) -> ItemStatus
        self._transformed = False

        self._length = None

    def is_cache_initialized(self) -> bool:
        return self._source is None

    @property
    def _is_unchanged_wrapper(self) -> bool:
        return self._source is not None and not self._updated_items

    def init_cache(self):
        if not self.is_cache_initialized():
            for _ in self._iter_init_cache(): pass

        self._length = len(self._storage)

    def _iter_init_cache(self) -> Iterable[DatasetItem]:
        # Merges the source and patch, caches the result and
        # provides an iterator for the resulting item sequence.
        #
        # If iterated in parallel, the result is undefined.
        # If storage is changed during iteration, the result is undefined.
        #
        # TODO: can potentially be optimized by sharing
        # the cache between parallel consumers and introducing some kind of lock

        patch = self._storage # must be empty after transforming
        cache = DatasetItemStorage()

        i = -1
        for i, item in enumerate(self._source):
            if item in cache:
                raise RepeatedItemError((item.id, item.subset))
            if item in patch:
                item = patch.get(item.id, item.subset)
            if self._updated_items.get((item.id, item.subset)) == \
                    ItemStatus.removed:
                item = None
            if item:
                cache.put(item)
                yield item
        if i == -1:
            cache = patch
            for item in patch:
                self._updated_items[(item.id, item.subset)] = ItemStatus.added
                yield item
        else:
            for item in patch:
                if item in cache: # already processed
                    continue
                self._updated_items[(item.id, item.subset)] = ItemStatus.added
                cache.put(item)
                yield item

        self._storage = cache
        source_cat = self._source.categories()
        if source_cat is not None:
            self._categories = source_cat
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
        elif self._source is not None:
            self.init_cache()
        return DatasetItemStorageDatasetView(self._storage, self._categories)

    def __len__(self) -> int:
        if self._length is None:
            self.init_cache()
        return self._length

    def categories(self) -> CategoriesInfo:
        if self._categories is not None:
            return self._categories
        else:
            return self._source.categories()

    def put(self, item):
        is_new = self._storage.put(item)

        if not self.is_cache_initialized() or not is_new:
            self._updated_items[(item.id, item.subset)] = ItemStatus.modified
        elif is_new:
            self._updated_items[(item.id, item.subset)] = ItemStatus.added

        if is_new and not self.is_cache_initialized():
            self._length = None
        if self._length is not None:
            self._length += is_new

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        item = self._storage.get(id, subset)
        if item is None and not self.is_cache_initialized():
            if self._source.get.__func__ == Extractor.get:
                # can be improved if IDataset is ABC
                self.init_cache()
                item = self._storage.get(id, subset)
            else:
                item = self._source.get(id, subset)
                if item:
                    self._storage.put(item)
        return item

    def remove(self, id, subset=None):
        id = str(id)
        subset = subset or DEFAULT_SUBSET_NAME

        self._storage.remove(id, subset)
        is_removed = self._updated_items.get((id, subset)) != ItemStatus.removed
        if is_removed:
            self._updated_items[(id, subset)] = ItemStatus.removed
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

    def transform(self, method, *args, **kwargs):
        self._source = method(self._merged(), *args, **kwargs)
        self._storage = DatasetItemStorage()
        # TODO: can be optimized by analyzing methods
        self._categories = None
        self._length = None
        self._transformed = True
        self._updated_items = {}

    def has_updated_items(self):
        return self._transformed or self._updated_items

    def get_patch(self):
        # Patch includes only added or modified items.
        # To find removed items, one needs to consult updated_items list.
        if self._transformed:
            self.init_cache()
            # Consider all items modified after transforming
            self._updated_items = {
                (item.id, item.subset): ItemStatus.modified
                for item in self._storage
            }
        return DatasetPatch(self._storage, self._categories,
            self._updated_items)

    def flush_changes(self):
        self._updated_items = {}
        self._transformed = False


class Dataset(IDataset):
    _global_eager = False

    @classmethod
    def from_iterable(cls, iterable: Iterable[DatasetItem],
            categories: Union[CategoriesInfo, List[str]] = None,
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
    def from_extractors(*sources: IDataset,
            env: Environment = None) -> 'Dataset':
        if len(sources) == 1:
            source = sources[0]
        else:
            from datumaro.components.operations import ExactMerge
            source = ExactMerge.merge(*sources)
            categories = ExactMerge.merge_categories(
                s.categories() for s in sources)
            source = DatasetItemStorageDatasetView(source, categories)

        return Dataset(source=source, env=env)

    def __init__(self, source: IDataset = None,
            categories: CategoriesInfo = None, env: Environment = None):
        super().__init__()

        assert env is None or isinstance(env, Environment), env
        self._env = env

        self.eager = None
        self._data = DatasetStorage(source, categories=categories)
        if self.is_eager:
            self.init_cache()

        self._format = DEFAULT_FORMAT
        self._source_path = None

    def define_categories(self, categories: Dict):
        assert not self._data._categories and self._data._source is None
        self._data._categories = categories

    def init_cache(self):
        self._data.init_cache()

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

    def __contains__(self, x: Union[DatasetItem, str, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        elif not isinstance(x, (tuple, list)):
            x = [x]
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
            remove_empty: bool = False) -> 'Dataset':
        if filter_annotations:
            return self.transform(XPathAnnotationsFilter, expr, remove_empty)
        else:
            return self.transform(XPathDatasetFilter, expr)

    def update(self, items: Iterable[DatasetItem]) -> 'Dataset':
        for item in items:
            self.put(item)
        return self

    def transform(self, method: Union[str, Transform],
            *args, **kwargs) -> 'Dataset':
        if isinstance(method, str):
            method = self.env.make_transform(method)

        self._data.transform(method, *args, **kwargs)
        if self.is_eager:
            self.init_cache()

        return self

    def run_model(self, model, batch_size=1) -> 'Dataset':
        from datumaro.components.launcher import Launcher, ModelTransform
        if isinstance(model, Launcher):
            return self.transform(ModelTransform, launcher=model,
                batch_size=batch_size)
        elif isinstance(model, ModelTransform):
            return self.transform(model, batch_size=batch_size)
        else:
            raise TypeError('Unexpected model argument type: %s' % type(model))

    def select(self, pred):
        class _DatasetFilter(Transform):
            def __iter__(self):
                return filter(pred, iter(self._extractor))

        return self.transform(_DatasetFilter)

    @property
    def data_path(self) -> Optional[str]:
        return self._source_path

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def is_modified(self) -> bool:
        return self._data.has_updated_items()

    @property
    def patch(self) -> DatasetPatch:
        return self._data.get_patch()

    @property
    def env(self) -> Environment:
        if not self._env:
            self._env = Environment()
        return self._env

    @property
    def is_cache_initialized(self) -> bool:
        return self._data.is_cache_initialized()

    @property
    def is_eager(self) -> bool:
        return self.eager if self.eager is not None else self._global_eager

    @property
    def is_bound(self) -> bool:
        return self._source_path and self._format

    def bind(self, path: str, format: str = None):
        self._source_path = path
        self._format = format or DEFAULT_FORMAT

    def flush_changes(self):
        self._data.flush_changes()

    @error_rollback('on_error', implicit=True)
    def export(self, save_dir: str, format, **kwargs):
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
            if not self.is_bound:
                self.bind(save_dir, format)
                self.flush_changes()
        else:
            converter.patch(self, self.patch, save_dir=save_dir, **kwargs)

    def save(self, save_dir: str = None, **kwargs):
        self.export(save_dir or self._source_path,
            format=self._format, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'Dataset':
        return cls.import_from(path, format=DEFAULT_FORMAT, **kwargs)

    @classmethod
    def import_from(cls, path: str, format: str = None, env: Environment = None,
            **kwargs) -> 'Dataset':
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
            raise DatumaroError("Unknown source format '%s'. To make it "
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
            raise DatumaroError(
                "Failed to detect dataset format automatically: "
                "no matching formats found")
        if 1 < len(matches):
            raise DatumaroError(
                "Failed to detect dataset format automatically:"
                " data matches more than one format: %s" % \
                ', '.join(matches))
        return matches[0]

@contextmanager
def eager_mode(new_mode=True, dataset: Dataset = None):
    if dataset is not None:
        old_mode = dataset.eager

        try:
            dataset.eager = new_mode
            yield
        finally:
            dataset.eager = old_mode
    else:
        old_mode = Dataset._global_eager

        try:
            Dataset._global_eager = new_mode
            yield
        finally:
            Dataset._global_eager = old_mode