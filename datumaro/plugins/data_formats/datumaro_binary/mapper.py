# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.util import dump_json, parse_json


class Mapper(ABC):
    @staticmethod
    @abstractmethod
    def forward(obj: Any) -> bytes:
        """Convert an object to bytes."""

    @staticmethod
    @abstractmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[Any, int]:
        """Build an object from bytes."""


class StringMapper(Mapper):
    @staticmethod
    def forward(obj: str) -> bytes:
        obj = obj.encode()
        length = len(obj)
        return struct.pack(f"I{length}s", length, obj)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[str, int]:
        length = struct.unpack_from("I", _bytes, offset)[0]
        offset += 4
        string = struct.unpack_from(f"{length}s", _bytes, offset)[0].decode()
        return string, offset + length


class DictMapper(Mapper):
    @staticmethod
    def forward(obj: Dict[str, Any]) -> bytes:
        msg = dump_json(obj)
        length = len(msg)
        return struct.pack(f"I{length}s", length, msg)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[Dict[str, Any], int]:
        length = struct.unpack_from("I", _bytes, offset)[0]
        offset += 4
        parsed_dict = parse_json(_bytes[offset : offset + length])
        return parsed_dict, offset + length


class MediaMapper(Mapper):
    ALLOWED_TYPES = {MediaType.IMAGE, MediaType.POINT_CLOUD}

    @staticmethod
    def forward(obj: MediaElement) -> bytes:
        if obj.MEDIA_TYPE == MediaType.IMAGE:
            size = obj.size if obj.has_size else (-1, -1)
        elif obj.MEDIA_TYPE == MediaType.POINT_CLOUD:
            size = (-1, -1)
        elif obj.MEDIA_TYPE == MediaType.UNKNOWN:
            size = (-1, -1)
        else:
            raise DatumaroError(f"{obj.MEDIA_TYPE} is not allowed for MediaMapper.")

        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack(f"Iii", obj.MEDIA_TYPE, size[0], size[1]))
        bytes_arr.extend(StringMapper.forward(obj.path))

        return bytes_arr

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[MediaElement, int]:
        media_type, height, width = struct.unpack_from("Iii", _bytes, offset)
        size = (height, width)
        offset += 12
        path, offset = StringMapper.backward(_bytes, offset)

        if media_type == MediaType.IMAGE:
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
        return bytes(bytes_arr)

    @staticmethod
    def backward(_bytes: bytes, offset: int = 0) -> Tuple[DatasetItem, int]:
        id, offset = StringMapper.backward(_bytes, offset)
        subset, offset = StringMapper.backward(_bytes, offset)
        media, offset = MediaMapper.backward(_bytes, offset)
        attributes, offset = DictMapper.backward(_bytes, offset)
        return DatasetItem(id=id, subset=subset, media=media, attributes=attributes), offset
