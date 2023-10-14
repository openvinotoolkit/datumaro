# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
import os
import struct
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType, PointCloud
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper
from datumaro.util.image import decode_image, encode_image, load_image
from .utils import b64decode, b64encode, pa_batches_decoder


@dataclass(frozen=True)
class CommonEntities:
    media_type: MediaType
    path: str
    attributes: Dict[str, Any]
    has_bytes: bool


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement], **options) -> Dict[str, Any]:
        if obj is None:
            return {"type": int(MediaType.NONE)}
        if obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj, **options)
        if obj._type == MediaType.POINT_CLOUD:
            return PointCloudMapper.forward(obj, **options)
        if obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(cls, idx: int, table: pa.Table) -> Optional[MediaElement]:
        media_type = MediaType(table.column("media_type")[idx].as_py())

        if media_type == MediaType.NONE:
            return None
        if media_type == MediaType.IMAGE:
            return ImageMapper.backward(idx, table)
        if media_type == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward(idx, table)
        if media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(idx, table)
        raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class MediaElementMapper(Mapper):
    MAGIC_PATH = "/NOT/A/REAL/PATH"
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> Dict[str, Any]:
        return {
            "type": int(obj.type),
            "path": getattr(obj, "path", cls.MAGIC_PATH),
            "has_bytes": False,
            "attributes": {},
        }

    @classmethod
    def backward_common(cls, idx: int, table: pa.Table) -> CommonEntities:
        media_type = table.column("media_type")[idx].as_py()
        assert media_type == cls.MEDIA_TYPE, f"Expect {cls.MEDIA_TYPE} but actual is {media_type}."
        attributes, _ = DictMapper.backward(table.column("media_attributes")[idx].as_py(), 0)

        return CommonEntities(
            media_type=media_type,
            path=table.column("media_path")[idx].as_py(),
            attributes=attributes,
            has_bytes=table.column("media_has_bytes")[idx].as_py(),
        )

    @classmethod
    def backward(cls, idx: int, table: pa.Table) -> MediaElement:
        _ = cls.backward_common(idx, table)
        return MediaElement()


class ImageMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.IMAGE
    AVAILABLE_SCHEMES = ("AS-IS", "PNG", "TIFF", "JPEG/95", "JPEG/75", "NONE")

    @classmethod
    def encode(cls, obj: Image, scheme: str = "PNG") -> Optional[bytes]:
        if scheme is None or scheme == "NONE":
            return None
        if scheme == "AS-IS":
            _bytes = obj.bytes
            if _bytes is not None:
                return _bytes
            # try to encode in PNG
            scheme = "PNG"

        options = {}
        if scheme.startswith("JPEG"):
            quality = int(scheme.split("/")[-1])
            options["ext"] = "JPEG"
            options["jpeg_quality"] = quality
        elif scheme == "PNG":
            options["ext"] = "PNG"
        elif scheme == "TIFF":
            options["ext"] = "TIFF"
        else:
            raise NotImplementedError

        data = obj.data
        if data is not None:
            return encode_image(obj.data, **options)
        return None

    @classmethod
    def decode(
        cls, path: Optional[str] = None, data: Optional[bytes] = None
    ) -> Optional[np.ndarray]:
        if path is None and data is None:
            return None
        if data is not None:
            return decode_image(data, np.uint8)
        if path is not None:
            return load_image(path, np.uint8)

    @classmethod
    def forward(
        cls, obj: Image, encoder: Union[str, Callable[[Image], bytes]] = "PNG"
    ) -> Dict[str, Any]:
        out = super().forward(obj)

        _bytes = None
        if isinstance(encoder, Callable):
            _bytes = encoder(obj)
        else:
            _bytes = cls.encode(obj, scheme=encoder)
        out["bytes"] = _bytes
        out["has_bytes"] = _bytes is not None
        out["attributes"] = DictMapper.forward(dict(size=obj.size))

        return out

    @classmethod
    def backward_from_dict(cls, obj: Dict[str, Any]) -> Image:
        media_dict = obj

        path = media_dict["path"]
        attributes, _ = DictMapper.backward(media_dict["attributes"], 0)

        _bytes = None
        if media_dict.get("bytes", None) is not None:
            _bytes = media_dict["bytes"]

        if _bytes:
            return Image.from_bytes(data=_bytes, size=attributes["size"])
        return Image.from_file(path=path, size=attributes["size"])

    @classmethod
    def backward(cls, idx: int, table: pa.Table) -> Image:
        common_entities = cls.backward_common(idx, table)

        if common_entities.has_bytes:
            return Image.from_bytes(
                data=lambda: table.column("media_bytes")[idx].as_py(),
                size=common_entities.attributes["size"],
            )

        return Image.from_file(path=common_entities.path, size=common_entities.attributes["size"])


# TODO: share binary for extra images
class PointCloudMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.POINT_CLOUD
    B64_PREFIX = "//B64_ENCODED//"

    @classmethod
    def forward(
        cls, obj: PointCloud, encoder: Union[str, Callable[[PointCloud], bytes]] = "PNG"
    ) -> Dict[str, Any]:
        out = super().forward(obj)

        _bytes = None
        if isinstance(encoder, Callable):
            _bytes = encoder(obj)
        elif encoder != "NONE":
            _bytes = obj.data
        out["bytes"] = _bytes

        out["attributes"] = DictMapper.forward(
            {
                str(idx): ImageMapper.forward(img, encoder=encoder)
                for idx, img in enumerate(obj.extra_images)
            }
        )

        return out

    @classmethod
    def backward(cls, idx: int, table: pa.Table) -> PointCloud:
        common_entities = cls.backward_common(idx, table)

        path = common_entities.path

        extra_images = [
            ImageMapper.backward_from_dict(v) for v in common_entities.attributes.values()
        ]

        if common_entities.has_bytes:
            return PointCloud.from_bytes(
                data=lambda: table.column("media_bytes")[idx].as_py(),
                extra_images=extra_images,
            )

        return PointCloud.from_file(path=path, extra_images=extra_images)
