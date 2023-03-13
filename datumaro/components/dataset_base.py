# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Type, TypeVar, Union, cast

import attr
import numpy as np
from attr import attrs, field

from datumaro.components.annotation import Annotation, AnnotationType, Categories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.importer import ImportContext, NullImportContext
from datumaro.components.media import Image, MediaElement, PointCloud
from datumaro.util.attrs_util import default_if_none, not_empty
from datumaro.util.definitions import DEFAULT_SUBSET_NAME

T = TypeVar("T", bound=MediaElement)


@attrs(order=False, init=False, slots=True)
class DatasetItem:
    id: str = field(converter=lambda x: str(x).replace("\\", "/"), validator=not_empty)

    subset: str = field(converter=lambda v: v or DEFAULT_SUBSET_NAME, default=None)

    media: Optional[MediaElement] = field(
        default=None, validator=attr.validators.optional(attr.validators.instance_of(MediaElement))
    )

    annotations: List[Annotation] = field(factory=list, validator=default_if_none(list))

    attributes: Dict[str, Any] = field(factory=dict, validator=default_if_none(dict))

    def wrap(item, **kwargs):
        return attr.evolve(item, **kwargs)

    def media_as(self, t: Type[T]) -> T:
        assert issubclass(t, MediaElement)
        return cast(t, self.media)

    def __init__(
        self,
        id: str,
        *,
        subset: Optional[str] = None,
        media: Union[str, MediaElement, None] = None,
        annotations: Optional[List[Annotation]] = None,
        attributes: Dict[str, Any] = None,
        image=None,
        point_cloud=None,
        related_images=None,
    ):
        if image is not None:
            warnings.warn(
                "'image' is deprecated and will be " "removed in future. Use 'media' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if isinstance(image, str):
                image = Image(path=image)
            elif isinstance(image, np.ndarray) or callable(image):
                image = Image(data=image)
            assert isinstance(image, Image)
            media = image
        elif point_cloud is not None:
            warnings.warn(
                "'point_cloud' is deprecated and will be "
                "removed in future. Use 'media' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if related_images is not None:
                warnings.warn(
                    "'related_images' is deprecated and will be "
                    "removed in future. Use 'media' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if isinstance(point_cloud, str):
                point_cloud = PointCloud(path=point_cloud, extra_images=related_images)
            assert isinstance(point_cloud, PointCloud)
            media = point_cloud

        self.__attrs_init__(
            id=id, subset=subset, media=media, annotations=annotations, attributes=attributes
        )

    # Deprecated. Provided for backward compatibility.
    @property
    def image(self) -> Optional[Image]:
        warnings.warn(
            "'DatasetItem.image' is deprecated and will be "
            "removed in future. Use '.media' and '.media_as()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(self.media, Image):
            return None
        return self.media_as(Image)

    # Deprecated. Provided for backward compatibility.
    @property
    def point_cloud(self) -> Optional[str]:
        warnings.warn(
            "'DatasetItem.point_cloud' is deprecated and will be "
            "removed in future. Use '.media' and '.media_as()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(self.media, PointCloud):
            return None
        return self.media_as(PointCloud).path

    # Deprecated. Provided for backward compatibility.
    @property
    def related_images(self) -> List[Image]:
        warnings.warn(
            "'DatasetItem.related_images' is deprecated and will be "
            "removed in future. Use '.media' and '.media_as()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(self.media, PointCloud):
            return []
        return self.media_as(PointCloud).extra_images

    # Deprecated. Provided for backward compatibility.
    @property
    def has_image(self):
        warnings.warn(
            "'DatasetItem.has_image' is deprecated and will be "
            "removed in future. Use '.media' and '.media_as()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return isinstance(self.media, Image)

    # Deprecated. Provided for backward compatibility.
    @property
    def has_point_cloud(self):
        warnings.warn(
            "'DatasetItem.has_point_cloud' is deprecated and will be "
            "removed in future. Use '.media' and '.media_as()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return isinstance(self.media, PointCloud)


DatasetInfo = Dict[str, Any]
CategoriesInfo = Dict[AnnotationType, Categories]


class IDataset:
    def __iter__(self) -> Iterator[DatasetItem]:
        """
        Provides sequential access to dataset items.
        """
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __bool__(self):  # avoid __len__ use for truth checking
        return True

    def subsets(self) -> Dict[str, IDataset]:
        """
        Enumerates subsets in the dataset. Each subset can be a dataset itself.
        """
        raise NotImplementedError()

    def get_subset(self, name) -> IDataset:
        raise NotImplementedError()

    def infos(self) -> DatasetInfo:
        """
        Returns meta-info of dataset.
        """
        raise NotImplementedError()

    def categories(self) -> CategoriesInfo:
        """
        Returns metainfo about dataset labels.
        """
        raise NotImplementedError()

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        """
        Provides random access to dataset items.
        """
        raise NotImplementedError()

    def media_type(self) -> Type[MediaElement]:
        """
        Returns media type of the dataset items.

        All the items are supposed to have the same media type.
        Supposed to be constant and known immediately after the
        object construction (i.e. doesn't require dataset iteration).
        """
        raise NotImplementedError()


class _DatasetBase(IDataset):
    def __init__(self, *, length: Optional[int] = None, subsets: Optional[Sequence[str]] = None):
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

    def subsets(self) -> Dict[str, IDataset]:
        if self._subsets is None:
            self._init_cache()
        return {name or DEFAULT_SUBSET_NAME: self.get_subset(name) for name in self._subsets}

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
            raise KeyError(
                "Unknown subset '%s', available subsets: %s" % (name, set(self._subsets))
            )

    def transform(self, method, *args, **kwargs):
        return method(self, *args, **kwargs)

    def select(self, pred):
        class _DatasetFilter(_DatasetBase):
            def __iter__(_):
                return filter(pred, iter(self))

            def infos(_):
                return self.infos()

            def categories(_):
                return self.categories()

            def media_type(_):
                return self.media_type()

        return _DatasetFilter()

    def infos(self):
        return {}

    def categories(self):
        return {}

    def get(self, id, subset=None):
        subset = subset or DEFAULT_SUBSET_NAME
        for item in self:
            if item.id == id and item.subset == subset:
                return item
        return None


class DatasetBase(_DatasetBase, CliPlugin):
    """
    A base class for user-defined and built-in extractors.
    Should be used in cases, where SubsetBase is not enough,
    or its use makes problems with performance, implementation etc.
    """

    def __init__(
        self,
        *,
        length: Optional[int] = None,
        subsets: Optional[Sequence[str]] = None,
        media_type: Type[MediaElement] = Image,
        ctx: Optional[ImportContext] = None,
    ):
        super().__init__(length=length, subsets=subsets)

        self._ctx: ImportContext = ctx or NullImportContext()
        self._media_type = media_type

    def media_type(self):
        return self._media_type


class SubsetBase(DatasetBase):
    """
    A base class for simple, single-subset extractors.
    Should be used by default for user-defined extractors.
    """

    def __init__(
        self,
        *,
        length: Optional[int] = None,
        subset: Optional[str] = None,
        media_type: Type[MediaElement] = Image,
        ctx: Optional[ImportContext] = None,
    ):
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(length=length, subsets=[self._subset], media_type=media_type, ctx=ctx)

        self._infos = {}
        self._categories = {}
        self._items = []

    def infos(self):
        return self._infos

    def categories(self):
        return self._categories

    def __iter__(self):
        yield from self._items

    def __len__(self):
        return len(self._items)

    def get(self, id, subset=None):
        assert subset == self._subset, "%s != %s" % (subset, self._subset)
        return super().get(id, subset or self._subset)
