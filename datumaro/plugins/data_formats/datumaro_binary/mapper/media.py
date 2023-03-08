# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from typing import Dict, Optional, Tuple

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud

from .common import Mapper, StringMapper


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement]) -> bytes:
        if obj is None:
            return struct.pack(f"<I", MediaType.NONE)
        elif obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj)
        elif obj._type == MediaType.POINT_CLOUD:
            return PointCloudMapper.forward(obj)
        elif obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        else:
            raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Optional[MediaElement], int]:
        (media_type,) = struct.unpack_from("<I", _bytes, offset)

        if media_type == MediaType.NONE:
            return None, offset + 4
        elif media_type == MediaType.IMAGE:
            return ImageMapper.backward(_bytes, offset)
        elif media_type == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward(_bytes, offset)
        elif media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(_bytes, offset)
        else:
            raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class MediaElementMapper(Mapper):
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack(f"<I", obj.type))
        bytes_arr.extend(StringMapper.forward(obj.path))

        return bytes(bytes_arr)

    @classmethod
    def backward_dict(cls, _bytes: bytes, offset: int = 0) -> Tuple[Dict, int]:
        (media_type,) = struct.unpack_from("<I", _bytes, offset)
        assert media_type == cls.MEDIA_TYPE, f"Expect {cls.MEDIA_TYPE} but actual is {media_type}."
        offset += 4
        path, offset = StringMapper.backward(_bytes, offset)
        return {"type": media_type, "path": path}, offset

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[MediaElement, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset)
        return MediaElement(path=media_dict["path"]), offset


class ImageMapper(MediaElementMapper):
    MAGIC_SIZE_FOR_NONE = (-1583, -1597)
    MEDIA_TYPE = MediaType.IMAGE

    @classmethod
    def forward(cls, obj: Image) -> bytes:
        size = obj.size if obj.has_size else cls.MAGIC_SIZE_FOR_NONE

        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack(f"<ii", size[0], size[1]))

        return bytes(bytes_arr)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[Image, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset)
        height, width = struct.unpack_from("<ii", _bytes, offset)
        size = (height, width)
        offset += 8
        return (
            Image(path=media_dict["path"], size=size if size != cls.MAGIC_SIZE_FOR_NONE else None),
            offset,
        )


class PointCloudMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.POINT_CLOUD

    @classmethod
    def forward(cls, obj: PointCloud) -> bytes:
        bytes_arr = bytearray()
        bytes_arr.extend(super().forward(obj))
        bytes_arr.extend(struct.pack(f"<I", len(obj.extra_images)))
        for img in obj.extra_images:
            bytes_arr.extend(ImageMapper.forward(img))

        return bytes(bytes_arr)

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Tuple[PointCloud, int]:
        media_dict, offset = cls.backward_dict(_bytes, offset)
        (len_extra_images,) = struct.unpack_from("<I", _bytes, offset)
        offset += 4

        extra_images = []
        for _ in range(len_extra_images):
            img, offset = ImageMapper.backward(_bytes, offset)
            extra_images.append(img)

        return PointCloud(media_dict["path"], extra_images), offset
