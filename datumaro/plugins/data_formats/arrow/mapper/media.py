# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import struct
from functools import partial
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper
from datumaro.util.image import decode_image, encode_image, load_image


class ImageFileMapper:
    AVAILABLE_SCHEMES = ("JPEG/75", "JPEG/95", "PNG", "TIFF", "AS-IS", "NONE")

    @classmethod
    def forward(cls, path, scheme: str = "JPEG/75") -> Optional[bytes]:
        if scheme == "NONE":
            return None
        elif scheme == "AS-IS":
            encoded = open(path, "rb").read()
            return encoded

        kwargs = {}
        if scheme.startswith("JPEG"):
            quality = int(scheme.split("/")[-1])
            kwargs["ext"] = "JPEG"
            kwargs["jpeg_quality"] = quality
        elif scheme == "PNG":
            kwargs["ext"] = "PNG"
        elif scheme == "TIFF":
            kwargs["ext"] = "TIFF"
        else:
            raise NotImplementedError

        image = load_image(path, np.uint8)
        encoded = encode_image(image, **kwargs)
        return encoded

    @classmethod
    def backward(cls, *, path: Optional[str] = None, data: Optional[bytes] = None) -> np.ndarray:
        assert (path is not None) ^ (data is not None), "Either one of path or data must be given"

        if data is not None:
            return decode_image(data, np.uint8)
        else:
            assert path is not None
            return load_image(path, np.uint8)


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement], **options) -> Dict[str, Any]:
        if obj is None:
            return struct.pack(f"<I", MediaType.NONE)
        elif obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj, **options)
        #  elif obj._type == MediaType.POINT_CLOUD:
        #      return PointCloudMapper.forward(obj)
        elif obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        else:
            raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(cls, _bytes: bytes, offset: int = 0) -> Optional[MediaElement]:
        (media_type,) = struct.unpack_from("<I", _bytes, offset)

        if media_type == MediaType.NONE:
            return None
        elif media_type == MediaType.IMAGE:
            return ImageMapper.backward(_bytes, offset)
        #  elif media_type == MediaType.POINT_CLOUD:
        #      return PointCloudMapper.backward(_bytes, offset)
        elif media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(_bytes, offset)
        else:
            raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")


class MediaElementMapper(Mapper):
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> Dict[str, Any]:
        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack(f"<I", obj.type))
        return {
            "type": bytes(bytes_arr),
            "path": obj.path,
        }

    @classmethod
    def backward_dict(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        obj = obj.copy()
        (media_type,) = struct.unpack_from("<I", obj["type"], 0)
        assert media_type == cls.MEDIA_TYPE, f"Expect {cls.MEDIA_TYPE} but actual is {media_type}."
        obj["type"] = media_type
        return obj

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> MediaElement:
        media_dict = cls.backward_dict(obj)
        return MediaElement(path=media_dict["path"])


class ImageMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.IMAGE

    @classmethod
    def forward(
        cls, obj: Image, encoder: Union[str, Callable[[str], bytes]] = "JPEG/75"
    ) -> Dict[str, Any]:
        out = super().forward(obj)
        if isinstance(encoder, Callable):
            _bytes = encoder(out["path"])
        else:
            _bytes = ImageFileMapper.forward(out["path"], encoder)
        out["bytes"] = _bytes

        out["attributes"] = DictMapper.forward(dict(size=obj.size))

        return out

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> Image:
        media_dict = cls.backward_dict(obj)

        path = media_dict["path"]
        attributes, _ = DictMapper.backward(media_dict["attributes"], 0)
        image_decoder = None
        if "bytes" in media_dict:
            image_decoder = partial(ImageFileMapper.backward, data=media_dict["bytes"])
        return Image(data=image_decoder, path=path, size=attributes["size"])
