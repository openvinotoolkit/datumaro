# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from typing import Optional, Tuple

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.plugins.data_formats.datumaro_binary.mapper.annotation import AnnotationListMapper

from .common import DictMapper, Mapper, StringMapper

# from .annotations import AnnotationListMapper


class MediaMapper(Mapper):
    MAGIC_SIZE_FOR_NONE = (-1, -1)

    @classmethod
    def forward(cls, obj: Optional[MediaElement]) -> bytes:
        if obj is None:
            size = cls.MAGIC_SIZE_FOR_NONE
        elif obj.MEDIA_TYPE == MediaType.IMAGE:
            size = obj.size if obj.has_size else cls.MAGIC_SIZE_FOR_NONE
        elif obj.MEDIA_TYPE == MediaType.POINT_CLOUD:
            size = cls.MAGIC_SIZE_FOR_NONE
        elif obj.MEDIA_TYPE == MediaType.UNKNOWN:
            size = cls.MAGIC_SIZE_FOR_NONE
        else:
            raise DatumaroError(f"{obj.MEDIA_TYPE} is not allowed for MediaMapper.")

        media_type = getattr(obj, "MEDIA_TYPE", MediaType.NONE)
        path = getattr(obj, "_path", "")

        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack(f"Iii", media_type, size[0], size[1]))
        bytes_arr.extend(StringMapper.forward(path))

        return bytes_arr

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Optional[MediaElement], int]:
        media_type, height, width = struct.unpack_from("Iii", _bytes, offset)
        size = (height, width)
        offset += 12
        path, offset = StringMapper.backward(_bytes, offset)

        if media_type == MediaType.NONE:
            return None, offset
        elif media_type == MediaType.IMAGE:
            return Image(path=path, size=size if size != (-1, -1) else None), offset
        elif media_type == MediaType.POINT_CLOUD:
            raise NotImplementedError
        elif media_type == MediaType.UNKNOWN:
            return MediaElement(path=path), offset
        else:
            raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class DatasetItemMapper(Mapper):
    @staticmethod
    def forward(obj: DatasetItem) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(StringMapper.forward(obj.id))
        bytes_arr.extend(StringMapper.forward(obj.subset))
        bytes_arr.extend(MediaMapper.forward(obj.media))
        bytes_arr.extend(DictMapper.forward(obj.attributes))
        bytes_arr.extend(AnnotationListMapper.forward(obj.annotations))
        return bytes(bytes_arr)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[DatasetItem, int]:
        id, offset = StringMapper.backward(_bytes, offset)
        subset, offset = StringMapper.backward(_bytes, offset)
        media, offset = MediaMapper.backward(_bytes, offset)
        attributes, offset = DictMapper.backward(_bytes, offset)
        annotations, offset = AnnotationListMapper.backward(_bytes, offset)
        return (
            DatasetItem(
                id=id, subset=subset, media=media, attributes=attributes, annotations=annotations
            ),
            offset,
        )
