# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import Enum, auto
from glob import iglob
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, NoReturn, Optional, TypeVar,
    Union,
)
import math
import os
import os.path as osp

from attr import attrs, define, field
import attr
import numpy as np

from datumaro.components.annotation import (
    Annotation, AnnotationType, Categories,
)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import (
    AnnotationImportError, DatasetNotFoundError, DatumaroError, ItemImportError,
)
from datumaro.components.format_detection import (
    FormatDetectionConfidence, FormatDetectionContext,
)
from datumaro.components.media import Image
from datumaro.util import is_method_redefined
from datumaro.util.attrs_util import default_if_none, not_empty

DEFAULT_SUBSET_NAME = 'default'

@attrs(slots=True, order=False)
class DatasetItem:
    id: str = field(converter=lambda x: str(x).replace('\\', '/'),
        validator=not_empty)
    annotations: List[Annotation] = field(
        factory=list, validator=default_if_none(list))
    subset: str = field(converter=lambda v: v or DEFAULT_SUBSET_NAME,
        default=None)

    # TODO: introduce "media" field with type info. Replace image and pcd.
    image: Optional[Image] = field(default=None)
    # TODO: introduce pcd type like Image
    point_cloud: Optional[str] = field(
        converter=lambda x: str(x).replace('\\', '/') if x else None,
        default=None)
    related_images: List[Image] = field(default=None)

    def __attrs_post_init__(self):
        if (self.has_image and self.has_point_cloud):
            raise ValueError("Can't set both image and point cloud info")
        if self.related_images and not self.has_point_cloud:
            raise ValueError("Related images require point cloud")

    def _image_converter(image):
        if callable(image) or isinstance(image, np.ndarray):
            image = Image(data=image)
        elif isinstance(image, str):
            image = Image(path=image)
        assert image is None or isinstance(image, Image), type(image)
        return image
    image.converter = _image_converter

    def _related_image_converter(images):
        return list(map(__class__._image_converter, images or []))
    related_images.converter = _related_image_converter

    @point_cloud.validator
    def _point_cloud_validator(self, attribute, pcd):
        assert pcd is None or isinstance(pcd, str), type(pcd)

    attributes: Dict[str, Any] = field(
        factory=dict, validator=default_if_none(dict))

    @property
    def has_image(self):
        return self.image is not None

    @property
    def has_point_cloud(self):
        return self.point_cloud is not None

    def wrap(item, **kwargs):
        return attr.evolve(item, **kwargs)


CategoriesInfo = Dict[AnnotationType, Categories]

class IExtractor:
    def __iter__(self) -> Iterator[DatasetItem]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __bool__(self): # avoid __len__ use for truth checking
        return True

    def subsets(self) -> Dict[str, IExtractor]:
        raise NotImplementedError()

    def get_subset(self, name) -> IExtractor:
        raise NotImplementedError()

    def categories(self) -> CategoriesInfo:
        raise NotImplementedError()

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        raise NotImplementedError()

class _ExtractorBase(IExtractor):
    def __init__(self, *, length=None, subsets=None):
        self._length = length
        self._subsets = subsets

    def _init_cache(self):
        subsets = set()
        length = -1
        for length, item in enumerate(self):
            subsets.add(item.subset)
        length += 1

        if self._length is None:
            self._length = length
        if self._subsets is None:
            self._subsets = subsets

    def __len__(self):
        if self._length is None:
            self._init_cache()
        return self._length

    def subsets(self) -> Dict[str, IExtractor]:
        if self._subsets is None:
            self._init_cache()
        return {name or DEFAULT_SUBSET_NAME: self.get_subset(name)
            for name in self._subsets}

    def get_subset(self, name):
        if self._subsets is None:
            self._init_cache()
        if name in self._subsets:
            if len(self._subsets) == 1:
                return self

            subset = self.select(lambda item: item.subset == name)
            subset._subsets = [name]
            return subset
        else:
            raise KeyError("Unknown subset '%s', available subsets: %s" % \
                (name, set(self._subsets)))

    def transform(self, method, *args, **kwargs):
        return method(self, *args, **kwargs)

    def select(self, pred):
        class _DatasetFilter(_ExtractorBase):
            def __iter__(_):
                return filter(pred, iter(self))
            def categories(_):
                return self.categories()

        return _DatasetFilter()

    def categories(self):
        return {}

    def get(self, id, subset=None):
        subset = subset or DEFAULT_SUBSET_NAME
        for item in self:
            if item.id == id and item.subset == subset:
                return item
        return None

T = TypeVar('T')

class _ImportFail(DatumaroError):
    pass

class ItemErrorAction(Enum):
    skip_item = auto()

class AnnotationErrorAction(Enum):
    skip_item = auto()
    skip_annotation = auto()

class ProgressReporter:
    def get_frequency(self) -> float:
        raise NotImplementedError

    def start(self, total: int, *, desc: Optional[str] = None):
        raise NotImplementedError

    def report_status(self, progress: int):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError

    def iter(self, iterable: Iterable[T], *,
            total: Optional[int] = None,
            desc: Optional[str]
    ) -> Iterable[T]:
        if total is None:
            if hasattr(iterable, '__len__'):
                total = len(iterable)

        self.start(total, desc=desc)

        if total:
            display_step = math.ceil(total * self.get_frequency())
        else:
            display_step = None
        for i, elem in enumerate(iterable):
            if not total or i % display_step == 0:
                self.report_status(i)

            yield elem

        self.report_status(i)

        self.finish()

class ErrorPolicy:
    def report_item_error(self,
            error: ItemImportError
    ) -> Union[ItemErrorAction, NoReturn]:
        raise NotImplementedError

    def report_annotation_error(self,
            error: AnnotationImportError
    ) -> Union[AnnotationErrorAction, NoReturn]:
        raise NotImplementedError

    def fail(self, error: Exception) -> NoReturn:
        raise _ImportFail from error

@define(eq=False)
class ImportContext:
    progress_reporter: Optional[ProgressReporter] = None
    error_policy: Optional[ErrorPolicy] = None

class Extractor(_ExtractorBase, CliPlugin):
    """
    A base class for user-defined and built-in extractors.
    Should be used in cases, where SourceExtractor is not enough,
    or its use makes problems with performance, implementation etc.
    """

    def __init__(self, *, length=None, subsets=None,
            ctx: Optional[ImportContext] = None):
        super().__init__(length=length, subsets=subsets)
        self._ctx = ctx

    def _with_progress(self, iterable: Iterable[T], *,
            total: Optional[int] = None,
            desc: Optional[str] = None
    ) -> Iterable[T]:
        if self._ctx and self._ctx.progress_reporter:
            yield from self._ctx.progress_reporter.iter(iterable,
                total=total, desc=desc)
        else:
            yield from iterable

    def _report_item_error(self, error: ItemImportError):
        if self._ctx and self._ctx.error_policy:
            return self._ctx.error_policy.report_annotation_error(error)
        raise _ImportFail from error

    def _report_annotation_error(self, error: AnnotationImportError):
        if self._ctx and self._ctx.error_policy:
            return self._ctx.error_policy.report_item_error(error)
        raise _ImportFail from error

class SourceExtractor(Extractor):
    """
    A base class for simple, single-subset extractors.
    Should be used by default for user-defined extractors.
    """

    def __init__(self, *, length=None, subset=None,
            ctx: Optional[ImportContext] = None):
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(length=length, subsets=[self._subset], ctx=ctx)

        self._categories = {}
        self._items = []

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items

    def __len__(self):
        return len(self._items)

    def get(self, id, subset=None):
        assert subset == self._subset, '%s != %s' % (subset, self._subset)
        return super().get(id, subset or self._subset)

class Importer(CliPlugin):
    @classmethod
    def detect(
        cls, context: FormatDetectionContext,
    ) -> Optional[FormatDetectionConfidence]:
        if not cls.find_sources_with_params(context.root_path):
            context.fail("specific requirement information unavailable")

        return FormatDetectionConfidence.LOW

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        raise NotImplementedError()

    @classmethod
    def find_sources_with_params(cls, path, **extra_params) -> List[Dict]:
        return cls.find_sources(path)

    def __call__(self, path, **extra_params):
        if not path or not osp.exists(path):
            raise DatasetNotFoundError(path)

        found_sources = self.find_sources_with_params(
            osp.normpath(path), **extra_params)
        if not found_sources:
            raise DatasetNotFoundError(path)

        sources = []
        for desc in found_sources:
            params = dict(extra_params)
            params.update(desc.get('options', {}))
            desc['options'] = params
            sources.append(desc)

        return sources

    @classmethod
    def _find_sources_recursive(cls, path: str, ext: Optional[str],
            extractor_name: str, filename: str = '*', dirname: str = '',
            file_filter: Optional[Callable[[str], bool]] = None,
            max_depth: int = 3):
        """
        Finds sources in the specified location, using the matching pattern
        to filter file names and directories.
        Supposed to be used, and to be the only call in subclasses.

        Parameters:
            path: a directory or file path, where sources need to be found.
            ext: file extension to match. To match directories,
                set this parameter to None or ''. Comparison is case-independent,
                a starting dot is not required.
            extractor_name: the name of the associated Extractor type
            filename: a glob pattern for file names
            dirname: a glob pattern for filename prefixes
            file_filter: a callable (abspath: str) -> bool, to filter paths found
            max_depth: the maximum depth for recursive search.

        Returns: a list of source configurations
            (i.e. Extractor type names and c-tor parameters)
        """

        if ext:
            if not ext.startswith('.'):
                ext = '.' + ext
            ext = ext.lower()

        if (path.lower().endswith(ext) and osp.isfile(path)) or \
                (not ext and dirname and osp.isdir(path) and \
                os.sep + osp.normpath(dirname.lower()) + os.sep in \
                    osp.abspath(path.lower()) + os.sep):
            sources = [{'url': path, 'format': extractor_name}]
        else:
            sources = []
            for d in range(max_depth + 1):
                sources.extend({'url': p, 'format': extractor_name} for p in
                    iglob(osp.join(path, *('*' * d), dirname, filename + ext))
                    if (callable(file_filter) and file_filter(p)) \
                    or (not callable(file_filter)))
                if sources:
                    break
        return sources

class Transform(_ExtractorBase, CliPlugin):
    """
    A base class for dataset transformations that change dataset items
    or their annotations.
    """

    @staticmethod
    def wrap_item(item, **kwargs):
        return item.wrap(**kwargs)

    def __init__(self, extractor: IExtractor):
        super().__init__()

        self._extractor = extractor

    def categories(self):
        return self._extractor.categories()

    def subsets(self):
        if self._subsets is None:
            self._subsets = set(self._extractor.subsets())
        return super().subsets()

    def __len__(self):
        assert self._length in {None, 'parent'} or isinstance(self._length, int)
        if self._length is None and \
                    not is_method_redefined('__iter__', Transform, self) \
                or self._length == 'parent':
            self._length = len(self._extractor)
        return super().__len__()

class ItemTransform(Transform):
    def transform_item(self, item: DatasetItem) -> Optional[DatasetItem]:
        """
        Returns a modified copy of the input item.

        Avoid changing and returning the input item, because it can lead to
        unexpected problems. Use wrap_item() or item.wrap() to simplify copying.
        """

        raise NotImplementedError()

    def __iter__(self):
        for item in self._extractor:
            item = self.transform_item(item)
            if item is not None:
                yield item
