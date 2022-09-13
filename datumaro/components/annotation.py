# Copyright (C) 2021-2022 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from enum import Enum, auto
from functools import partial
from itertools import zip_longest
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

import attr
import numpy as np
from attr import asdict, attrs, field
from typing_extensions import Literal

from datumaro.components.media import Image
from datumaro.util.attrs_util import default_if_none, not_empty


class AnnotationType(Enum):
    label = auto()
    mask = auto()
    points = auto()
    polygon = auto()
    polyline = auto()
    bbox = auto()
    caption = auto()
    cuboid_3d = auto()
    super_resolution_annotation = auto()
    depth_annotation = auto()
    skeleton = auto()


COORDINATE_ROUNDING_DIGITS = 2

NO_GROUP = 0


@attrs(slots=True, kw_only=True, order=False)
class Annotation:
    """
    A base annotation class.

    Derived classes must define the '_type' class variable with a value
    from the AnnotationType enum.
    """

    # Describes an identifier of the annotation
    # Is not required to be unique within DatasetItem annotations or dataset
    id: int = field(default=0, validator=default_if_none(int))

    # Arbitrary annotation-specific attributes. Typically, includes
    # metainfo and properties that are not covered by other fields.
    # If possible, try to limit value types of values by the simple
    # builtin types (int, float, bool, str) to increase compatibility with
    # different formats.
    # There are some established names for common attributes like:
    # - "occluded" (bool)
    # - "visible" (bool)
    # Possible dataset attributes can be described in Categories.attributes.
    attributes: Dict[str, Any] = field(factory=dict, validator=default_if_none(dict))

    # Annotations can be grouped, which means they describe parts of a
    # single object. The value of 0 means there is no group.
    group: int = field(default=NO_GROUP, validator=default_if_none(int))

    @property
    def type(self) -> AnnotationType:
        return self._type  # must be set in subclasses

    def as_dict(self) -> Dict[str, Any]:
        "Returns a dictionary { field_name: value }"
        return asdict(self)

    def wrap(self, **kwargs):
        "Returns a modified copy of the object"
        return attr.evolve(self, **kwargs)


@attrs(slots=True, kw_only=True, order=False)
class Categories:
    """
    A base class for annotation metainfo. It is supposed to include
    dataset-wide metainfo like available labels, label colors,
    label attributes etc.
    """

    # Describes the list of possible annotation-type specific attributes
    # in a dataset.
    attributes: Set[str] = field(factory=set, validator=default_if_none(set), eq=False)


@attrs(slots=True, order=False)
class LabelCategories(Categories):
    @attrs(slots=True, order=False)
    class Category:
        name: str = field(converter=str, validator=not_empty)
        parent: str = field(default="", validator=default_if_none(str))
        attributes: Set[str] = field(factory=set, validator=default_if_none(set))

    items: List[str] = field(factory=list, validator=default_if_none(list))
    _indices: Dict[str, int] = field(factory=dict, init=False, eq=False)

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[
            Union[
                str,
                Tuple[str],
                Tuple[str, str],
                Tuple[str, str, List[str]],
            ]
        ],
    ) -> LabelCategories:
        """
        Creates a LabelCategories from iterable.

        Args:
            iterable: This iterable object can be:

                - a list of str - will be interpreted as list of Category names
                - a list of positional arguments - will generate Categories
                  with these arguments

        Returns: a LabelCategories object
        """

        temp_categories = cls()

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

    def add(
        self, name: str, parent: Optional[str] = None, attributes: Optional[Set[str]] = None
    ) -> int:
        assert name
        assert name not in self._indices, name

        index = len(self.items)
        self.items.append(self.Category(name, parent, attributes))
        self._indices[name] = index
        return index

    def find(self, name: str) -> Tuple[Optional[int], Optional[Category]]:
        index = self._indices.get(name)
        if index is not None:
            return index, self.items[index]
        return index, None

    def __getitem__(self, idx: int) -> Category:
        return self.items[idx]

    def __contains__(self, value: Union[int, str]) -> bool:
        if isinstance(value, str):
            return self.find(value)[1] is not None
        else:
            return 0 <= value and value < len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Category]:
        return iter(self.items)


@attrs(slots=True, order=False)
class Label(Annotation):
    _type = AnnotationType.label
    label: int = field(converter=int)


RgbColor = Tuple[int, int, int]

Colormap = Dict[int, RgbColor]
"""Represents { index -> color } mapping for segmentation masks"""


@attrs(slots=True, eq=False, order=False)
class MaskCategories(Categories):
    """
    Describes a color map for segmentation masks.
    """

    @classmethod
    def generate(cls, size: int = 255, include_background: bool = True) -> MaskCategories:
        """
        Generates MaskCategories with the specified size.

        If include_background is True, the result will include the item
            "0: (0, 0, 0)", which is typically used as a background color.
        """
        from datumaro.util.mask_tools import generate_colormap

        return cls(generate_colormap(size, include_background=include_background))

    colormap: Colormap = field(factory=dict, validator=default_if_none(dict))
    _inverse_colormap: Optional[Dict[RgbColor, int]] = field(
        default=None, validator=attr.validators.optional(dict)
    )

    @property
    def inverse_colormap(self) -> Dict[RgbColor, int]:
        from datumaro.util.mask_tools import invert_colormap

        if self._inverse_colormap is None:
            if self.colormap is not None:
                self._inverse_colormap = invert_colormap(self.colormap)
        return self._inverse_colormap

    def __contains__(self, idx: int) -> bool:
        return idx in self.colormap

    def __getitem__(self, idx: int) -> RgbColor:
        return self.colormap[idx]

    def __len__(self) -> int:
        return len(self.colormap)

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


BinaryMaskImage = np.ndarray  # 2d array of type bool
IndexMaskImage = np.ndarray  # 2d array of type int


@attrs(slots=True, eq=False, order=False)
class Mask(Annotation):
    """
    Represents a 2d single-instance binary segmentation mask.
    """

    _type = AnnotationType.mask
    _image = field()
    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )
    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)

    def __attrs_post_init__(self):
        if isinstance(self._image, np.ndarray):
            self._image = self._image.astype(bool)

    @property
    def image(self) -> BinaryMaskImage:
        image = self._image
        if callable(image):
            image = image()
        return image

    def as_class_mask(self, label_id: Optional[int] = None) -> IndexMaskImage:
        """
        Produces a class index mask. Mask label id can be changed.
        """
        if label_id is None:
            label_id = self.label
        from datumaro.util.mask_tools import make_index_mask

        return make_index_mask(self.image, label_id)

    def as_instance_mask(self, instance_id: int) -> IndexMaskImage:
        """
        Produces a instance index mask.
        """
        from datumaro.util.mask_tools import make_index_mask

        return make_index_mask(self.image, instance_id)

    def get_area(self) -> int:
        return np.count_nonzero(self.image)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """
        Computes the bounding box of the mask.

        Returns: [x, y, w, h]
        """
        from datumaro.util.mask_tools import find_mask_bbox

        return find_mask_bbox(self.image)

    def paint(self, colormap: Colormap) -> np.ndarray:
        """
        Applies a colormap to the mask and produces the resulting image.
        """
        from datumaro.util.mask_tools import paint_mask

        return paint_mask(self.as_class_mask(), colormap)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if not isinstance(other, __class__):
            return False
        return (
            (self.label == other.label)
            and (self.z_order == other.z_order)
            and (np.array_equal(self.image, other.image))
        )


@attrs(slots=True, eq=False, order=False)
class RleMask(Mask):
    """
    An RLE-encoded instance segmentation mask.
    """

    _rle = field()  # uses pycocotools RLE representation

    _image = field(init=False, default=None)

    @property
    def image(self) -> BinaryMaskImage:
        return self._decode(self.rle)

    @property
    def rle(self):
        rle = self._rle
        if callable(rle):
            rle = rle()
        return rle

    @staticmethod
    def _decode(rle):
        from pycocotools import mask as mask_utils

        return mask_utils.decode(rle)

    def get_area(self) -> int:
        from pycocotools import mask as mask_utils

        return mask_utils.area(self.rle)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        from pycocotools import mask as mask_utils

        return mask_utils.toBbox(self.rle)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return super().__eq__(other)
        return self.rle == other.rle


CompiledMaskImage = np.ndarray  # 2d of integers (of different precision)


class CompiledMask:
    """
    Represents class- and instance- segmentation masks with
    all the instances (opposed to single-instance masks).
    """

    @staticmethod
    def from_instance_masks(
        instance_masks: Iterable[Mask],
        instance_ids: Optional[Iterable[int]] = None,
        instance_labels: Optional[Iterable[int]] = None,
    ) -> CompiledMask:
        """
        Joins instance masks into a single mask. Masks are sorted by
        z_order (ascending) prior to merging.

        Parameters:
            instance_ids: Instance id values for the produced instance mask.
                By default, mask positions are used.
            instance_labels: Instance label id values for the produced class
                mask. By default, mask labels are used.
        """

        from datumaro.util.mask_tools import make_index_mask

        instance_ids = instance_ids or []
        instance_labels = instance_labels or []
        masks = sorted(
            zip_longest(instance_masks, instance_ids, instance_labels), key=lambda m: m[0].z_order
        )

        max_index = len(masks) + 1
        index_dtype = np.min_scalar_type(max_index)

        masks = (
            (m, 1 + i, id if id is not None else 1 + i, label if label is not None else m.label)
            for i, (m, id, label) in enumerate(masks)
        )

        # This optimized version is supposed for:
        # 1. Avoiding memory explosion on materialization of all masks
        # 2. Optimizing mask materialization calls (RLE decoding)
        # 3. Optimizing intermediate mask memory use
        #
        # Basically, a mask can be quite large (e.g. 10k x 10k @ int32 etc.),
        # so we can only afford having just few copies in
        # memory simultaneously.

        it = iter(masks)

        instance_map = [0]
        class_map = [0]

        m, idx, instance_id, class_id = next(it)
        if not class_id:
            idx = 0
        index_mask = make_index_mask(m.image, idx, dtype=index_dtype)
        instance_map.append(instance_id)
        class_map.append(class_id)

        for m, idx, instance_id, class_id in it:
            if not class_id:
                idx = 0
            index_mask = np.where(m.image, idx, index_mask)
            instance_map.append(instance_id)
            class_map.append(class_id)

        # Generate compiled masks

        if np.array_equal(instance_map, range(max_index)):
            merged_instance_mask = index_mask
        else:
            merged_instance_mask = np.array(instance_map, dtype=np.min_scalar_type(instance_map))[
                index_mask
            ]

        merged_class_mask = np.array(class_map, dtype=np.min_scalar_type(class_map))[index_mask]

        return __class__(class_mask=merged_class_mask, instance_mask=merged_instance_mask)

    def __init__(
        self,
        class_mask: Union[None, CompiledMaskImage, Callable[[], CompiledMaskImage]] = None,
        instance_mask: Union[None, CompiledMaskImage, Callable[[], CompiledMaskImage]] = None,
    ):
        self._class_mask = class_mask
        self._instance_mask = instance_mask

    @staticmethod
    def _get_image(image):
        if callable(image):
            return image()
        return image

    @property
    def class_mask(self) -> Optional[CompiledMaskImage]:
        return self._get_image(self._class_mask)

    @property
    def instance_mask(self) -> Optional[CompiledMaskImage]:
        return self._get_image(self._instance_mask)

    @property
    def instance_count(self) -> int:
        return int(self.instance_mask.max())

    def get_instance_labels(self) -> Dict[int, int]:
        """
        Matches the class and instance masks.

        Returns: { instance id: class id }
        """

        class_shift = 16
        m = (self.class_mask.astype(np.uint32) << class_shift) + self.instance_mask.astype(
            np.uint32
        )
        keys = np.unique(m)
        instance_labels = {
            int(k & ((1 << class_shift) - 1)): int(k >> class_shift)
            for k in keys
            if k & ((1 << class_shift) - 1) != 0
        }
        return instance_labels

    def extract(self, instance_id: int) -> IndexMaskImage:
        """
        Extracts a single-instance mask from the compiled mask.
        """

        return self.instance_mask == instance_id

    def lazy_extract(self, instance_id: int) -> Callable[[], IndexMaskImage]:
        return partial(self.extract, instance_id)


@attrs(slots=True, order=False)
class _Shape(Annotation):
    # Flattened list of point coordinates
    points: List[float] = field(
        converter=lambda x: np.around(x, COORDINATE_ROUNDING_DIGITS).tolist(), factory=list
    )

    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )

    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)

    def get_area(self):
        raise NotImplementedError()

    def get_bbox(self) -> Tuple[float, float, float, float]:
        "Returns [x, y, w, h]"

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


@attrs(slots=True, order=False)
class PolyLine(_Shape):
    _type = AnnotationType.polyline

    def as_polygon(self):
        return self.points[:]

    def get_area(self):
        return 0


@attrs(slots=True, init=False, order=False)
class Cuboid3d(Annotation):
    _type = AnnotationType.cuboid_3d
    _points: List[float] = field(default=None)
    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )

    @_points.validator
    def _points_validator(self, attribute, points):
        if points is None:
            points = [0, 0, 0, 0, 0, 0, 1, 1, 1]
        else:
            assert len(points) == 3 + 3 + 3, points
            points = np.around(points, COORDINATE_ROUNDING_DIGITS).tolist()
        self._points = points

    def __init__(self, position, rotation=None, scale=None, **kwargs):
        assert len(position) == 3, position
        if not rotation:
            rotation = [0] * 3
        if not scale:
            scale = [1] * 3
        kwargs.pop("points", None)
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
        self.position[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()

    @property
    def rotation(self):
        """[rx, ry, rz]"""
        return self._points[3:6]

    @rotation.setter
    def _set_rotation(self, value):
        self.rotation[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()

    @property
    def scale(self):
        """[sx, sy, sz]"""
        return self._points[6:9]

    @scale.setter
    def _set_scale(self, value):
        self.scale[:] = np.around(value, COORDINATE_ROUNDING_DIGITS).tolist()


@attrs(slots=True, order=False)
class Polygon(_Shape):
    _type = AnnotationType.polygon

    def __attrs_post_init__(self):
        # keep the message on a single line to produce informative output
        assert len(self.points) % 2 == 0 and 3 <= len(self.points) // 2, (
            "Wrong polygon points: %s" % self.points
        )

    def get_area(self):
        import pycocotools.mask as mask_utils

        x, y, w, h = self.get_bbox()
        rle = mask_utils.frPyObjects([self.points], y + h, x + w)
        area = mask_utils.area(rle)[0]
        return area


@attrs(slots=True, init=False, order=False)
class Bbox(_Shape):
    _type = AnnotationType.bbox

    def __init__(self, x, y, w, h, *args, **kwargs):
        kwargs.pop("points", None)  # comes from wrap()
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
        return [x, y, x + w, y, x + w, y + h, x, y + h]

    def iou(self, other: _Shape) -> Union[float, Literal[-1]]:
        from datumaro.util.annotation_util import bbox_iou

        return bbox_iou(self.get_bbox(), other.get_bbox())

    def wrap(item, **kwargs):
        d = {"x": item.x, "y": item.y, "w": item.w, "h": item.h}
        d.update(kwargs)
        return attr.evolve(item, **d)


@attrs(slots=True, order=False)
class PointsCategories(Categories):
    """
    Describes (key-)point metainfo such as point names and joints.
    """

    @attrs(slots=True, order=False)
    class Category:
        # Names for specific points, e.g. eye, hose, mouth etc.
        # These labels are not required to be in LabelCategories
        labels: List[str] = field(factory=list, validator=default_if_none(list))

        # Pairs of connected point indices
        joints: Set[Tuple[int, int]] = field(factory=set, validator=default_if_none(set))

    items: Dict[int, Category] = field(factory=dict, validator=default_if_none(dict))

    @classmethod
    def from_iterable(
        cls,
        iterable: Union[
            Tuple[int, List[str]],
            Tuple[int, List[str], Set[Tuple[int, int]]],
        ],
    ) -> PointsCategories:
        """
        Create PointsCategories from an iterable.

        Args:
            iterable: An Iterable with the following elements:

                - a label id
                - a list of positional arguments for Categories

        Returns:
            PointsCategories: PointsCategories object
        """
        temp_categories = cls()

        for args in iterable:
            temp_categories.add(*args)
        return temp_categories

    def add(
        self,
        label_id: int,
        labels: Optional[Iterable[str]] = None,
        joints: Iterable[Tuple[int, int]] = None,
    ):
        if joints is None:
            joints = []
        joints = set(map(tuple, joints))
        self.items[label_id] = self.Category(labels, joints)

    def __contains__(self, idx: int) -> bool:
        return idx in self.items

    def __getitem__(self, idx: int) -> Category:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)


@attrs(slots=True, order=False)
class Points(_Shape):
    """
    Represents an ordered set of points.
    """

    class Visibility(Enum):
        absent = 0
        hidden = 1
        visible = 2

    _type = AnnotationType.points

    visibility: List[bool] = field(default=None)

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
        assert len(self.points) % 2 == 0, self.points

    def get_area(self):
        return 0

    def get_bbox(self):
        xs = [
            p
            for p, v in zip(self.points[0::2], self.visibility)
            if v != __class__.Visibility.absent
        ]
        ys = [
            p
            for p, v in zip(self.points[1::2], self.visibility)
            if v != __class__.Visibility.absent
        ]
        x0 = min(xs, default=0)
        x1 = max(xs, default=0)
        y0 = min(ys, default=0)
        y1 = max(ys, default=0)
        return [x0, y0, x1 - x0, y1 - y0]


@attrs(slots=True, order=False)
class Caption(Annotation):
    """
    Represents arbitrary text annotations.
    """

    _type = AnnotationType.caption
    caption: str = field(converter=str)


@attrs(slots=True, order=False)
class _ImageAnnotation(Annotation):
    image: Image = field()


@attrs(slots=True, order=False)
class SuperResolutionAnnotation(_ImageAnnotation):
    """
    Represents high resolution images.
    """

    _type = AnnotationType.super_resolution_annotation


@attrs(slots=True, order=False)
class DepthAnnotation(_ImageAnnotation):
    """
    Represents depth images.
    """

    _type = AnnotationType.depth_annotation


@attrs(slots=True, order=False)
class Skeleton(Annotation):
    """
    Represents a skeleton.
    """

    _type = AnnotationType.skeleton

    elements: List[Points] = field(factory=list)

    label: Optional[int] = field(
        converter=attr.converters.optional(int), default=None, kw_only=True
    )

    z_order: int = field(default=0, validator=default_if_none(int), kw_only=True)

    def __attrs_post_init__(self):
        pass

    def get_area(self):
        return 0

    def get_bbox(self):
        xs = []
        ys = []
        for element in self.elements:
            if (
                element.type is not AnnotationType.points
                or element.type is AnnotationType.points
                and [v for v in element.visibility if v != element.Visibility.absent]
            ):
                bbox = element.get_bbox()
                xs.extend([bbox[0], bbox[2] + bbox[0]])
                ys.extend([bbox[1], bbox[3] + bbox[1]])

        x0 = min(xs, default=0)
        x1 = max(xs, default=0)
        y0 = min(ys, default=0)
        y1 = max(ys, default=0)
        return [x0, y0, x1 - x0, y1 - y0]
