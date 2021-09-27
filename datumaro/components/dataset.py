# Copyright (C) 2020-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from contextlib import contextmanager
from copy import copy
from enum import Enum, auto
from typing import (
    Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union,
)
import inspect
import logging as log
import os
import os.path as osp

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_filter import (
    XPathAnnotationsFilter, XPathDatasetFilter,
)
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    CategoriesRedefinedError, ConflictingCategoriesError,
    MultipleFormatsMatchError, NoMatchingFormatsError, RepeatedItemError,
    UnknownFormatError,
)
from datumaro.components.extractor import (
    DEFAULT_SUBSET_NAME, CategoriesInfo, DatasetItem, Extractor, IExtractor,
    ItemTransform, Transform,
)
from datumaro.plugins.transforms import ReorderLabels
from datumaro.util import is_method_redefined
from datumaro.util.log_utils import logging_disabled
from datumaro.util.os_util import rmtree
from datumaro.util.scope import on_error_do, scoped

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

    def is_empty(self) -> bool:
        # Subsets might contain removed items, so this may differ from __len__
        return all(len(s) == 0 for s in self.data.values())

    def put(self, item: DatasetItem) -> bool:
        subset = self.data.setdefault(item.subset, {})
        is_new = subset.get(item.id) is None
        self._traversal_order[(item.id, item.subset)] = item
        subset[item.id] = item
        return is_new

    def get(self, id: Union[str, DatasetItem], subset: Optional[str] = None,
            dummy: Any = None) -> Optional[DatasetItem]:
        if isinstance(id, DatasetItem):
            id, subset = id.id, id.subset
        else:
            id = str(id)
            subset = subset or DEFAULT_SUBSET_NAME

        return self.data.get(subset, {}).get(id, dummy)

    def remove(self, id: Union[str, DatasetItem],
            subset: Optional[str] = None) -> bool:
        if isinstance(id, DatasetItem):
            id, subset = id.id, id.subset
        else:
            id = str(id)
            subset = subset or DEFAULT_SUBSET_NAME

        subset_data = self.data.setdefault(subset, {})
        is_removed = subset_data.get(id) is not None
        subset_data[id] = None
        if is_removed:
            self._traversal_order.pop((id, subset))
        return is_removed

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        if not isinstance(x, tuple):
            x = [x]
        dummy = 0
        return self.get(*x, dummy=dummy) is not dummy

    def get_subset(self, name):
        return self.data.get(name, {})

    def subsets(self):
        return self.data

    def __copy__(self):
        copied = DatasetItemStorage()
        copied._traversal_order = copy(self._traversal_order)
        copied.data = copy(self.data)
        return copied

class DatasetItemStorageDatasetView(IDataset):
    class Subset(IDataset):
        def __init__(self, parent: 'DatasetItemStorageDatasetView', name: str):
            super().__init__()
            self.parent = parent
            self.name = name

        @property
        def _data(self):
            return self.parent._get_subset_data(self.name)

        def __iter__(self):
            for item in self._data.values():
                if item:
                    yield item

        def __len__(self):
            return len(self._data)

        def put(self, item):
            return self._data.put(item)

        def get(self, id, subset=None):
            assert (subset or DEFAULT_SUBSET_NAME) == \
                   (self.name or DEFAULT_SUBSET_NAME)
            return self._data.get(id, subset)

        def remove(self, id, subset=None):
            assert (subset or DEFAULT_SUBSET_NAME) == \
                   (self.name or DEFAULT_SUBSET_NAME)
            return self._data.remove(id, subset)

        def get_subset(self, name):
            assert (name or DEFAULT_SUBSET_NAME) == \
                   (self.name or DEFAULT_SUBSET_NAME)
            return self

        def subsets(self):
            return { self.name or DEFAULT_SUBSET_NAME : self }

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


class ItemStatus(Enum):
    added = auto()
    modified = auto()
    removed = auto()

class DatasetPatch:
    class DatasetPatchWrapper(DatasetItemStorageDatasetView):
        # The purpose of this class is to indicate that the input dataset is
        # a patch and autofill patch info in Converter
        def __init__(self, patch: 'DatasetPatch', parent: IDataset):
            super().__init__(patch.data, parent.categories())
            self.patch = patch

        def subsets(self):
            return { s: self.get_subset(s) for s in self.patch.updated_subsets }

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
            self._updated_subsets = {s: ItemStatus.modified
                for s in self.data.subsets()}
        return self._updated_subsets

    def __contains__(self, x: Union[DatasetItem, Tuple[str, str]]) -> bool:
        return x in self.data

    def as_dataset(self, parent: IDataset) -> IDataset:
        return __class__.DatasetPatchWrapper(self, parent)

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
        assert (subset or DEFAULT_SUBSET_NAME) == \
               (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.get(id, subset=self.name)

    def remove(self, id, subset=None):
        assert (subset or DEFAULT_SUBSET_NAME) == \
               (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.remove(id, subset=self.name)

    def get_subset(self, name):
        assert (name or DEFAULT_SUBSET_NAME) == \
               (self.name or DEFAULT_SUBSET_NAME)
        return self

    def subsets(self):
        if (self.name or DEFAULT_SUBSET_NAME) == DEFAULT_SUBSET_NAME:
            return self.parent.subsets()
        return { self.name: self }

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

        # Possible combinations:
        # 1. source + storage
        #      - Storage contains a patch to the Source data.
        # 2. no source + storage
        #      - a dataset created from scratch
        #      - a dataset from a source or transform, which was cached
        self._source = source
        self._storage = DatasetItemStorage() # patch or cache
        self._transforms = [] # A stack of postponed transforms

        # Describes changes in the dataset since initialization
        self._updated_items = {} # (id, subset) -> ItemStatus

        self._flush_changes = False # Deferred flush indicator

        self._length = 0 if source is None else None

    def is_cache_initialized(self) -> bool:
        return self._source is None and not self._transforms

    @property
    def _is_unchanged_wrapper(self) -> bool:
        return self._source is not None and self._storage.is_empty() and \
            not self._transforms

    def init_cache(self):
        if not self.is_cache_initialized():
            for _ in self._iter_init_cache(): pass

    def _iter_init_cache(self) -> Iterable[DatasetItem]:
        # Merges the source, source transforms and patch, caches the result
        # and provides an iterator for the resulting item sequence.
        #
        # If iterated in parallel, the result is undefined.
        # If storage is changed during iteration, the result is undefined.
        #
        # TODO: can potentially be optimized by sharing
        # the cache between parallel consumers and introducing some kind of lock
        #
        # Cases:
        # 1. Has source and patch
        # 2. Has source, transforms and patch
        #   a. Transforms affect only an item (i.e. they are local)
        #   b. Transforms affect whole dataset
        #
        # The patch is always applied on top of the source / transforms stack.

        class _StackedTransform(Transform):
            def __init__(self, source, transforms):
                super().__init__(source)

                self.is_local = True
                self.transforms = []
                for transform in transforms:
                    source = transform[0](source, *transform[1], **transform[2])
                    self.transforms.append(source)

                    if self.is_local and not isinstance(source, ItemTransform):
                        self.is_local = False

            def transform_item(self, item):
                for t in self.transforms:
                    if item is None:
                        break
                    item = t.transform_item(item)
                return item

            def __iter__(self):
                yield from self.transforms[-1]

            def categories(self):
                return self.transforms[-1].categories()

        def _update_status(item_id, new_status: ItemStatus):
            current_status = self._updated_items.get(item_id)

            if current_status is None:
                self._updated_items[item_id] = new_status
            elif new_status == ItemStatus.removed:
                if current_status == ItemStatus.added:
                    self._updated_items.pop(item_id)
                else:
                    self._updated_items[item_id] = ItemStatus.removed
            elif new_status == ItemStatus.modified:
                if current_status != ItemStatus.added:
                    self._updated_items[item_id] = ItemStatus.modified
            elif new_status == ItemStatus.added:
                if current_status != ItemStatus.added:
                    self._updated_items[item_id] = ItemStatus.modified
            else:
                assert False, "Unknown status %s" % new_status

        patch = self._storage # must be empty after transforming
        cache = DatasetItemStorage()
        source = self._source
        transform = None

        if self._transforms:
            transform = _StackedTransform(source, self._transforms)
            if transform.is_local:
                # An optimized way to find modified items:
                # Transform items inplace and analyze transform outputs
                pass
            else:
                # A generic way to find modified items:
                # Collect all the dataset original ids and compare
                # with transform outputs.
                # TODO: introduce Extractor.items() / .ids() to avoid extra
                # dataset traversals?
                old_ids = set((item.id, item.subset) for item in source)
                source = transform

        i = -1
        for i, item in enumerate(source):
            if transform and transform.is_local:
                old_id = (item.id, item.subset)
                item = transform.transform_item(item)

            item_id = (item.id, item.subset) if item else None

            if item_id in cache:
                raise RepeatedItemError(item_id)

            if item in patch:
                # Apply changes from the patch
                item = patch.get(*item_id)
            elif transform and not self._flush_changes:
                # Find changes made by transforms, if not overridden by patch
                if transform.is_local:
                    if not item:
                        _update_status(old_id, ItemStatus.removed)
                    elif old_id != item_id:
                        _update_status(old_id, ItemStatus.removed)
                        _update_status(item_id, ItemStatus.added)
                    else:
                        # Consider all items modified without comparison,
                        # because such comparison would be very expensive
                        _update_status(old_id, ItemStatus.modified)
                else:
                    if item:
                        if item_id not in old_ids:
                            _update_status(item_id, ItemStatus.added)
                        else:
                            _update_status(item_id, ItemStatus.modified)

            if not item:
                continue

            cache.put(item)
            yield item

        if i == -1:
            cache = patch
            for item in patch:
                if not self._flush_changes:
                    _update_status((item.id, item.subset), ItemStatus.added)
                yield item
        else:
            for item in patch:
                if item in cache: # already processed
                    continue
                if not self._flush_changes:
                    _update_status((item.id, item.subset), ItemStatus.added)
                cache.put(item)
                yield item

        if not self._flush_changes and transform and not transform.is_local:
            # Mark removed items that were not produced by transforms
            for old_id in old_ids:
                if old_id not in self._updated_items:
                    self._updated_items[old_id] = ItemStatus.removed

        self._storage = cache
        self._length = len(cache)

        if transform:
            source_cat = transform.categories()
        else:
            source_cat = source.categories()
        if source_cat is not None:
            self._categories = source_cat

        self._source = None
        self._transforms = []

        if self._flush_changes:
            self._flush_changes = False
            self._updated_items = {}

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
        if self.is_cache_initialized():
            return self._categories
        elif self._categories is not None:
            return self._categories
        elif any(is_method_redefined('categories', Transform, t[0])
                for t in self._transforms):
            self.init_cache()
            return self._categories
        else:
            return self._source.categories()

    def define_categories(self, categories: CategoriesInfo):
        if self._categories or self._source is not None:
            raise CategoriesRedefinedError()
        self._categories = categories

    def put(self, item):
        is_new = self._storage.put(item)

        if not self.is_cache_initialized() or is_new:
            self._updated_items[(item.id, item.subset)] = ItemStatus.added
        else:
            self._updated_items[(item.id, item.subset)] = ItemStatus.modified

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
        # TODO: check if this can be optimized in case of transforms
        # and other cases
        return self._merged().subsets()

    def transform(self, method: Type[Transform], *args, **kwargs):
        # Flush accumulated changes
        if not self._storage.is_empty():
            source = self._merged()
            self._storage = DatasetItemStorage()
        else:
            source = self._source

        if not self._transforms:
            # The stack of transforms only needs a single source
            self._source = source
        self._transforms.append((method, args, kwargs))

        if is_method_redefined('categories', Transform, method):
            self._categories = None
        self._length = None

    def has_updated_items(self):
        return bool(self._transforms) or bool(self._updated_items)

    def get_patch(self):
        # Patch includes only added or modified items.
        # To find removed items, one needs to consult updated_items list.
        if self._transforms:
            self.init_cache()

        # The current patch (storage)
        # - can miss some removals done so we add them manually
        # - can include items than not in the patch
        #     (e.g. an item could get there after source was cached)
        # So we reconstruct the patch instead of copying storage.
        patch = DatasetItemStorage()
        for (item_id, subset), status in self._updated_items.items():
            if status is ItemStatus.removed:
                patch.remove(item_id, subset)
            else:
                patch.put(self._storage.get(item_id, subset))

        return DatasetPatch(patch, self._categories, self._updated_items)

    def flush_changes(self):
        self._updated_items = {}
        if not (self.is_cache_initialized() or self._is_unchanged_wrapper):
            self._flush_changes = True

    def update(self, source: Union[DatasetPatch, Iterable[DatasetItem]]):
        # TODO: provide a more efficient implementation with patch reuse

        if isinstance(source, DatasetPatch):
            if source.categories() != self.categories():
                raise ConflictingCategoriesError()

            for item_id, status in source.updated_items.items():
                if status == ItemStatus.removed:
                    self.remove(*item_id)
                else:
                    self.put(source.data.get(*item_id))
        elif isinstance(source, IExtractor):
            mapping = {}
            my_cat = self.categories().get(
                AnnotationType.label, LabelCategories())
            for patch_id, patch_label in enumerate(source.categories().get(
                    AnnotationType.label, LabelCategories())):
                my_id = my_cat.find(patch_label.name)[0]
                if my_id is None:
                    raise ConflictingCategoriesError()

                mapping[patch_id] = my_id

            for item in ReorderLabels(source, mapping):
                self.put(item)
        else:
            for item in source:
                self.put(item)

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
        self._options = {}

    def define_categories(self, categories: Dict):
        self._data.define_categories(categories)

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

    def update(self,
            source: Union[DatasetPatch, IExtractor, Iterable[DatasetItem]]) \
                -> 'Dataset':
        """
        Updates items of the current dataset from other dataset or an
        iterable (the source). Items from the source overwrite matching
        items in the current dataset. Unmatched items are just appended.

        If the source is a DatasetPatch, the removed items in the patch
        will be removed in the current dataset.

        If the source is a dataset, labels are matched. If the labels match,
        but the order is different, the annotation labels will be remapped to
        the current dataset label order during updating.
        """
        self._data.update(source)
        return self

    def transform(self, method: Union[str, Type[Transform]],
            *args, **kwargs) -> 'Dataset':
        """
        Applies some function to dataset items.
        """

        if isinstance(method, str):
            method = self.env.transforms[method]

        if inspect.isclass(method) and issubclass(method, Transform):
            pass
        else:
            raise TypeError("Unexpected 'method' argument type: %s" % \
                type(method))

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
            raise TypeError("Unexpected 'model' argument type: %s" % \
                type(model))

    def select(self, pred):
        class _DatasetFilter(ItemTransform):
            def transform_item(self, item):
                if pred(item):
                    return item
                return None

        return self.transform(_DatasetFilter)

    @property
    def data_path(self) -> Optional[str]:
        return self._source_path

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @property
    def is_modified(self) -> bool:
        return self._data.has_updated_items()

    def get_patch(self) -> DatasetPatch:
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
        return bool(self._source_path) and bool(self._format)

    def bind(self, path: str, format: Optional[str] = None, *,
            options: Optional[Dict[str, Any]] = None):
        """
        Binds the dataset to a speific directory.
        Allows to set default saving parameters.

        The following saves will be done to this directory by default and will
        use the saved parameters.
        """

        self._source_path = path
        self._format = format or DEFAULT_FORMAT
        self._options = options or {}

    def flush_changes(self):
        self._data.flush_changes()

    @scoped
    def export(self, save_dir: str, format, **kwargs):
        if not save_dir:
            raise ValueError("Dataset export path is not specified")

        inplace = (save_dir == self._source_path and format == self._format)

        if isinstance(format, str):
            converter = self.env.converters[format]
        else:
            converter = format

        save_dir = osp.abspath(save_dir)
        if not osp.exists(save_dir):
            on_error_do(rmtree, save_dir, ignore_errors=True)
            inplace = False
        os.makedirs(save_dir, exist_ok=True)

        if not inplace:
            converter.convert(self, save_dir=save_dir, **kwargs)
            if not self.is_bound:
                self.bind(save_dir, format, options=copy(kwargs))
                self.flush_changes()
        else:
            converter.patch(self, self.get_patch(), save_dir=save_dir, **kwargs)

    def save(self, save_dir: str = None, **kwargs):
        options = dict(self._options)
        options.update(kwargs)

        self.export(save_dir or self._source_path,
            format=self._format, **options)

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
                detected_sources = importer(path, **kwargs)
        elif format in env.extractors:
            detected_sources = [{
                'url': path, 'format': format, 'options': kwargs
            }]
        else:
            raise UnknownFormatError(format)

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
            raise NoMatchingFormatsError()
        if 1 < len(matches):
            raise MultipleFormatsMatchError(matches)
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
