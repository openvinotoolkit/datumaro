# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from glob import iglob
from typing import Callable, Dict, Iterable, List, Optional
import os
import os.path as osp

from attr import attrib, attrs
import attr
import numpy as np

from datumaro.components.annotation import AnnotationType, Categories
from datumaro.components.errors import DatasetNotFoundError
from datumaro.util import is_method_redefined
from datumaro.util.attrs_util import default_if_none, not_empty
from datumaro.util.image import Image

# Re-export some names from .annotation for backwards compatibility.
import datumaro.components.annotation # isort:skip
for _name in [
    'Annotation', 'AnnotationType', 'Bbox', 'Caption', 'Categories',
    'CompiledMask', 'Cuboid3d', 'Label', 'LabelCategories', 'Mask',
    'MaskCategories', 'Points', 'PointsCategories', 'Polygon', 'RleMask',
]:
    globals()[_name] = getattr(datumaro.components.annotation, _name)

DEFAULT_SUBSET_NAME = 'default'

@attrs
class DatasetItem:
    id = attrib(converter=lambda x: str(x).replace('\\', '/'),
        type=str, validator=not_empty)
    annotations = attrib(factory=list, validator=default_if_none(list))
    subset = attrib(converter=lambda v: v or DEFAULT_SUBSET_NAME,
        type=str, default=None)

    # Currently unused
    path = attrib(factory=list, validator=default_if_none(list))

    # TODO: introduce "media" field with type info. Replace image and pcd.
    image = attrib(type=Optional[Image], default=None)
    # TODO: introduce pcd type like Image
    point_cloud = attrib(converter=lambda x: \
            str(x).replace('\\', '/') if x else None,
        type=Optional[str], default=None)
    related_images = attrib(type=List[Image], default=None)

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

    attributes = attrib(factory=dict, validator=default_if_none(dict))

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
    def __iter__(self) -> Iterable[DatasetItem]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __bool__(self): # avoid __len__ use for truth checking
        return True

    def subsets(self) -> Dict[str, 'IExtractor']:
        raise NotImplementedError()

    def get_subset(self, name) -> 'IExtractor':
        raise NotImplementedError()

    def categories(self) -> CategoriesInfo:
        raise NotImplementedError()

    def get(self, id, subset=None) -> Optional[DatasetItem]:
        raise NotImplementedError()

class ExtractorBase(IExtractor):
    def __init__(self, length=None, subsets=None):
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
        class _DatasetFilter(ExtractorBase):
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

class Extractor(ExtractorBase):
    """
    A base class for user-defined and built-in extractors.
    Should be used in cases, where SourceExtractor is not enough,
    or its use makes problems with performance, implementation etc.
    """

class SourceExtractor(Extractor):
    """
    A base class for simple, single-subset extractors.
    Should be used by default for user-defined extractors.
    """

    def __init__(self, length=None, subset=None):
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(length=length, subsets=[self._subset])

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

class Importer:
    @classmethod
    def detect(cls, path):
        return len(cls.find_sources_with_params(path)) != 0

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        raise NotImplementedError()

    @classmethod
    def find_sources_with_params(cls, path, **extra_params) -> List[Dict]:
        return cls.find_sources(path)

    def __call__(self, path, **extra_params):
        from datumaro.components.project import Project  # cyclic import
        project = Project()

        sources = self.find_sources_with_params(osp.normpath(path), **extra_params)
        if len(sources) == 0:
            raise DatasetNotFoundError(path)

        for desc in sources:
            params = dict(extra_params)
            params.update(desc.get('options', {}))
            desc['options'] = params

            source_name = osp.splitext(osp.basename(desc['url']))[0]
            project.add_source(source_name, desc)

        return project

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
        - path - a directory or file path, where sources need to be found.
        - ext - file extension to match. To match directories,
            set this parameter to None or ''. Comparison is case-independent,
            a starting dot is not required.
        - extractor_name - the name of the associated Extractor type
        - filename - a glob pattern for file names
        - dirname - a glob pattern for filename prefixes
        - file_filter - a callable (abspath: str) -> bool, to filter paths found
        - max_depth - the maximum depth for recursive search.

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

class Transform(ExtractorBase):
    """
    A base class for dataset transformations that change dataset items
    or their annotations.
    """

    @staticmethod
    def wrap_item(item, **kwargs):
        return item.wrap(**kwargs)

    def __init__(self, extractor):
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
