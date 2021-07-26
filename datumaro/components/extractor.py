# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from glob import iglob
from typing import Callable, Dict, Iterable, List, Optional
import os
import os.path as osp

from attr import attrib, attrs
import attr
import numpy as np

from datumaro.util.attrs_util import default_if_none, not_empty
from datumaro.util.image import Image


class AnnotationType(Enum):
    label = auto()
    mask = auto()
    points = auto()
    polygon = auto()
    polyline = auto()
    bbox = auto()
    caption = auto()
    cuboid_3d = auto()

_COORDINATE_ROUNDING_DIGITS = 2

@attrs(kw_only=True)
class Annotation:
    id = attrib(default=0, validator=default_if_none(int))
    attributes = attrib(factory=dict, validator=default_if_none(dict))
    group = attrib(default=0, validator=default_if_none(int))

    def __attrs_post_init__(self):
        assert isinstance(self.type, AnnotationType)

    @property
    def type(self) -> AnnotationType:
        return self._type # must be set in subclasses

    def wrap(self, **kwargs):
        return attr.evolve(self, **kwargs)

@attrs(kw_only=True)
class Categories:
    attributes = attrib(factory=set, validator=default_if_none(set), eq=False)

@attrs
class LabelCategories(Categories):
    @attrs(repr_ns='LabelCategories')
    class Category:
        name = attrib(converter=str, validator=not_empty)
        parent = attrib(default='', validator=default_if_none(str))
        attributes = attrib(factory=set, validator=default_if_none(set))

    items = attrib(factory=list, validator=default_if_none(list))
    _indices = attrib(factory=dict, init=False, eq=False)

    @classmethod
    def from_iterable(cls, iterable):
        """Generation of LabelCategories from iterable object

        Args:
            iterable ([type]): This iterable object can be:
            1)simple str - will generate one Category with str as name
            2)list of str - will interpreted as list of Category names
            3)list of positional arguments - will generate Categories
            with this arguments


        Returns:
            LabelCategories: LabelCategories object
        """
        temp_categories = cls()

        if isinstance(iterable, str):
            iterable = [[iterable]]

        for category in iterable:
            if isinstance(category, str):
                category = [category]
            temp_categories.add(*category)

        return temp_categories

    def __attrs_post_init__(self):
        self._reindex()

    def _reindex(self):
        indices = {}
        for index, item in enumerate(self.items):
            assert item.name not in self._indices
            indices[item.name] = index
        self._indices = indices

    def add(self, name: str, parent: str = None, attributes: dict = None):
        assert name
        assert name not in self._indices, name

        index = len(self.items)
        self.items.append(self.Category(name, parent, attributes))
        self._indices[name] = index
        return index

    def find(self, name: str):
        index = self._indices.get(name)
        if index is not None:
            return index, self.items[index]
        return index, None

    def __getitem__(self, idx):
        return self.items[idx]

    def __contains__(self, idx):
        return 0 <= idx and idx < len(self.items)

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

@attrs
class Label(Annotation):
    _type = AnnotationType.label
    label = attrib(converter=int)

@attrs(eq=False)
class MaskCategories(Categories):
    @classmethod
    def generate(cls, size=255, include_background=True):
        """
        Generates a color map with the specified size.

        If include_background is True, the result will include the item
            "0: (0, 0, 0)", which is typically used as a background color.
        """
        from datumaro.util.mask_tools import generate_colormap
        colormap = generate_colormap(size + (not include_background))
        if not include_background:
            colormap.pop(0)
            colormap = { k - 1: v for k, v in colormap.items() }
        return cls(colormap)

    colormap = attrib(factory=dict, validator=default_if_none(dict))
    _inverse_colormap = attrib(default=None,
        validator=attr.validators.optional(dict))

    @property
    def inverse_colormap(self):
        from datumaro.util.mask_tools import invert_colormap
        if self._inverse_colormap is None:
            if self.colormap is not None:
                self._inverse_colormap = invert_colormap(self.colormap)
        return self._inverse_colormap

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not isinstance(other, __class__):
            return False
        for label_id, my_color in self.colormap.items():
            other_color = other.colormap.get(label_id)
            if not np.array_equal(my_color, other_color):
                return False
        return True

@attrs(eq=False)
class Mask(Annotation):
    _type = AnnotationType.mask
    _image = attrib()
    label = attrib(converter=attr.converters.optional(int),
        default=None, kw_only=True)
    z_order = attrib(default=0, validator=default_if_none(int), kw_only=True)

    def __attrs_post_init__(self):
        if isinstance(self._image, np.ndarray):
            self._image = self._image.astype(bool)

    @property
    def image(self):
        if callable(self._image):
            return self._image()
        return self._image

    def as_class_mask(self, label_id=None):
        if label_id is None:
            label_id = self.label
        from datumaro.util.mask_tools import make_index_mask
        return make_index_mask(self.image, label_id)

    def as_instance_mask(self, instance_id):
        from datumaro.util.mask_tools import make_index_mask
        return make_index_mask(self.image, instance_id)

    def get_area(self):
        return np.count_nonzero(self.image)

    def get_bbox(self):
        from datumaro.util.mask_tools import find_mask_bbox
        return find_mask_bbox(self.image)

    def paint(self, colormap):
        from datumaro.util.mask_tools import paint_mask
        return paint_mask(self.as_class_mask(), colormap)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not isinstance(other, __class__):
            return False
        return \
            (self.label == other.label) and \
            (self.z_order == other.z_order) and \
            (np.array_equal(self.image, other.image))

@attrs(eq=False)
class RleMask(Mask):
    rle = attrib()
    _image = attrib(default=attr.Factory(
        lambda self: self._lazy_decode(self.rle),
        takes_self=True), init=False)

    @staticmethod
    def _lazy_decode(rle):
        from pycocotools import mask as mask_utils
        return lambda: mask_utils.decode(rle)

    def get_area(self):
        from pycocotools import mask as mask_utils
        return mask_utils.area(self.rle)

    def get_bbox(self):
        from pycocotools import mask as mask_utils
        return mask_utils.toBbox(self.rle)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return super().__eq__(other)
        return self.rle == other.rle

class CompiledMask:
    @staticmethod
    def from_instance_masks(instance_masks,
            instance_ids=None, instance_labels=None, dtype=None):
        from datumaro.util.mask_tools import make_index_mask

        if instance_ids:
            assert len(instance_ids) == len(instance_masks)
        else:
            instance_ids = [None] * len(instance_masks)

        if instance_labels:
            assert len(instance_labels) == len(instance_masks)
        else:
            instance_labels = [None] * len(instance_masks)

        instance_masks = sorted(enumerate(instance_masks),
            key=lambda m: m[1].z_order)
        instance_masks = ((m.image, 1 + j,
                instance_ids[i] if instance_ids[i] is not None else 1 + j,
                instance_labels[i] if instance_labels[i] is not None else m.label
            ) for j, (i, m) in enumerate(instance_masks))

        # 1. Avoid memory explosion on materialization of all masks
        # 2. Optimize materialization calls
        it = iter(instance_masks)

        instance_map = [0]
        class_map = [0]

        m, idx, instance_id, class_id = next(it)
        if not class_id:
            idx = 0
        index_mask = make_index_mask(m, idx, dtype=dtype)
        instance_map.append(instance_id)
        class_map.append(class_id)

        for m, idx, instance_id, class_id in it:
            if not class_id:
                idx = 0
            index_mask = np.where(m, idx, index_mask)
            instance_map.append(instance_id)
            class_map.append(class_id)

        if np.array_equal(instance_map, range(idx + 1)):
            merged_instance_mask = index_mask
        else:
            merged_instance_mask = np.array(instance_map,
                dtype=np.min_scalar_type(instance_map))[index_mask]
        dtype_mask = dtype if dtype else np.min_scalar_type(class_map)
        merged_class_mask = np.array(class_map, dtype=dtype_mask)[index_mask]

        return __class__(class_mask=merged_class_mask,
            instance_mask=merged_instance_mask)

    def __init__(self, class_mask=None, instance_mask=None):
        self._class_mask = class_mask
        self._instance_mask = instance_mask

    @staticmethod
    def _get_image(image):
        if callable(image):
            return image()
        return image

    @property
    def class_mask(self):
        return self._get_image(self._class_mask)

    @property
    def instance_mask(self):
        return self._get_image(self._instance_mask)

    @property
    def instance_count(self):
        return int(self.instance_mask.max())

    def get_instance_labels(self):
        class_shift = 16
        m = (self.class_mask.astype(np.uint32) << class_shift) \
            + self.instance_mask.astype(np.uint32)
        keys = np.unique(m)
        instance_labels = {k & ((1 << class_shift) - 1): k >> class_shift
            for k in keys if k & ((1 << class_shift) - 1) != 0
        }
        return instance_labels

    def extract(self, instance_id):
        return self.instance_mask == instance_id

    def lazy_extract(self, instance_id):
        return lambda: self.extract(instance_id)

@attrs
class _Shape(Annotation):
    points = attrib(converter=lambda x:
        [round(p, _COORDINATE_ROUNDING_DIGITS) for p in x])
    label = attrib(converter=attr.converters.optional(int),
        default=None, kw_only=True)
    z_order = attrib(default=0, validator=default_if_none(int), kw_only=True)

    def get_area(self):
        raise NotImplementedError()

    def get_bbox(self):
        points = self.points
        if not points:
            return None

        xs = [p for p in points[0::2]]
        ys = [p for p in points[1::2]]
        x0 = min(xs)
        x1 = max(xs)
        y0 = min(ys)
        y1 = max(ys)
        return [x0, y0, x1 - x0, y1 - y0]

@attrs
class PolyLine(_Shape):
    _type = AnnotationType.polyline

    def as_polygon(self):
        return self.points[:]

    def get_area(self):
        return 0


@attrs(init=False)
class Cuboid3d(Annotation):
    _type = AnnotationType.cuboid_3d
    _points = attrib(type=list, default=None)
    label = attrib(converter=attr.converters.optional(int),
        default=None, kw_only=True)

    @_points.validator
    def _points_validator(self, attribute, points):
        if points is None:
            points = [0, 0, 0,  0, 0, 0,  1, 1, 1]
        else:
            assert len(points) == 3 + 3 + 3, points
            points = [round(p, _COORDINATE_ROUNDING_DIGITS) for p in points]
        self._points = points

    def __init__(self, position, rotation=None, scale=None, **kwargs):
        assert len(position) == 3, position
        if not rotation:
            rotation = [0] * 3
        if not scale:
            scale = [1] * 3
        kwargs.pop('points', None)
        self.__attrs_init__(points=[*position, *rotation, *scale], **kwargs)

    @property
    def position(self):
        """[x, y, z]"""
        return self._points[0:3]

    @position.setter
    def _set_poistion(self, value):
        # TODO: fix the issue with separate coordinate rounding:
        # self.position[0] = 12.345676
        # - the number assigned won't be rounded.
        self.position[:] = \
            [round(p, _COORDINATE_ROUNDING_DIGITS) for p in value]

    @property
    def rotation(self):
        """[rx, ry, rz]"""
        return self._points[3:6]

    @rotation.setter
    def _set_rotation(self, value):
        self.rotation[:] = \
            [round(p, _COORDINATE_ROUNDING_DIGITS) for p in value]

    @property
    def scale(self):
        """[sx, sy, sz]"""
        return self._points[6:9]

    @scale.setter
    def _set_scale(self, value):
        self.scale[:] = \
            [round(p, _COORDINATE_ROUNDING_DIGITS) for p in value]


@attrs
class Polygon(_Shape):
    _type = AnnotationType.polygon

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        # keep the message on a single line to produce informative output
        assert len(self.points) % 2 == 0 and 3 <= len(self.points) // 2, "Wrong polygon points: %s" % self.points

    def get_area(self):
        import pycocotools.mask as mask_utils

        x, y, w, h = self.get_bbox()
        rle = mask_utils.frPyObjects([self.points], y + h, x + w)
        area = mask_utils.area(rle)[0]
        return area

@attrs(init=False)
class Bbox(_Shape):
    _type = AnnotationType.bbox

    def __init__(self, x, y, w, h, *args, **kwargs):
        kwargs.pop('points', None) # comes from wrap()
        self.__attrs_init__([x, y, x + w, y + h], *args, **kwargs)

    @property
    def x(self):
        return self.points[0]

    @property
    def y(self):
        return self.points[1]

    @property
    def w(self):
        return self.points[2] - self.points[0]

    @property
    def h(self):
        return self.points[3] - self.points[1]

    def get_area(self):
        return self.w * self.h

    def get_bbox(self):
        return [self.x, self.y, self.w, self.h]

    def as_polygon(self):
        x, y, w, h = self.get_bbox()
        return [
            x, y,
            x + w, y,
            x + w, y + h,
            x, y + h
        ]

    def iou(self, other):
        from datumaro.util.annotation_util import bbox_iou
        return bbox_iou(self.get_bbox(), other.get_bbox())

    def wrap(item, **kwargs):
        d = {'x': item.x, 'y': item.y, 'w': item.w, 'h': item.h}
        d.update(kwargs)
        return attr.evolve(item, **d)


@attrs
class PointsCategories(Categories):
    @attrs(repr_ns="PointsCategories")
    class Category:
        labels = attrib(factory=list, validator=default_if_none(list))
        joints = attrib(factory=set, validator=default_if_none(set))

    items = attrib(factory=dict, validator=default_if_none(dict))

    @classmethod
    def from_iterable(cls, iterable):
        """Generation of PointsCategories from iterable object

        Args:
            iterable ([type]): This iterable object can be:
            1) list of positional arguments - will generate Categories
                with these arguments

        Returns:
            PointsCategories: PointsCategories object
        """
        temp_categories = cls()

        for category in iterable:
            temp_categories.add(*category)
        return temp_categories

    def add(self, label_id, labels=None, joints=None):
        if joints is None:
            joints = []
        joints = set(map(tuple, joints))
        self.items[label_id] = self.Category(labels, joints)

@attrs
class Points(_Shape):
    class Visibility(Enum):
        absent = 0
        hidden = 1
        visible = 2

    _type = AnnotationType.points

    visibility = attrib(type=list, default=None)
    @visibility.validator
    def _visibility_validator(self, attribute, visibility):
        if visibility is None:
            visibility = [self.Visibility.visible] * (len(self.points) // 2)
        else:
            for i, v in enumerate(visibility):
                if not isinstance(v, self.Visibility):
                    visibility[i] = self.Visibility(v)
        assert len(visibility) == len(self.points) // 2
        self.visibility = visibility

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        assert len(self.points) % 2 == 0, self.points

    def get_area(self):
        return 0

    def get_bbox(self):
        xs = [p for p, v in zip(self.points[0::2], self.visibility)
            if v != __class__.Visibility.absent]
        ys = [p for p, v in zip(self.points[1::2], self.visibility)
            if v != __class__.Visibility.absent]
        x0 = min(xs, default=0)
        x1 = max(xs, default=0)
        y0 = min(ys, default=0)
        y1 = max(ys, default=0)
        return [x0, y0, x1 - x0, y1 - y0]

@attrs
class Caption(Annotation):
    _type = AnnotationType.caption
    caption = attrib(converter=str)


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
            raise Exception("Unknown subset '%s', available subsets: %s" % \
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
        return len(cls.find_sources(path)) != 0

    @classmethod
    def find_sources(cls, path) -> List[Dict]:
        raise NotImplementedError()

    def __call__(self, path, **extra_params):
        found_sources = self.find_sources(osp.normpath(path))
        if len(found_sources) == 0:
            raise Exception("Failed to find dataset at '%s'" % path)

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

        Paramters:
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

        if (ext and path.lower().endswith(ext) and osp.isfile(path)) or \
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
                    self.__iter__.__func__ == __class__.__iter__ \
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
