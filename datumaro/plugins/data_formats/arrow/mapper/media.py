# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import struct
from functools import partial
from typing import Any, Callable, Dict, Optional, Union, List

import pyarrow as pa
import numpy as np

from datumaro.components.errors import DatumaroError
from datumaro.components.media import Image, MediaElement, MediaType
from datumaro.plugins.data_formats.datumaro_binary.mapper.common import DictMapper, Mapper
from datumaro.util.image import decode_image, encode_image, load_image

from .utils import pa_batches_decoder


class ImageFileMapper:
    AVAILABLE_SCHEMES = ("JPEG/75", "JPEG/95", "PNG", "TIFF", "AS-IS", "NONE")

    @classmethod
    def forward(
        cls,
        path: Optional[str] = None,
        data: Optional[np.ndarray] = None,
        scheme: str = "JPEG/75",
    ) -> Optional[bytes]:
        assert (path is not None) ^ (data is not None), "Either one of path or data must be given."

        if scheme == "NONE":
            return None
        elif scheme == "AS-IS":
            assert path is not None
            encoded = open(path, "rb").read()
            return encoded

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

        if path:
            data = load_image(path, np.uint8)
        assert data is not None
        encoded = encode_image(data, **options)
        return encoded

    @classmethod
    def backward(cls, *, path: Optional[str] = None, data: Optional[bytes] = None) -> Optional[np.ndarray]:
        if path is not None and data is not None:
            return None
        if data is not None:
            return decode_image(data, np.uint8)
        if path is not None:
            return load_image(path, np.uint8)


class MediaMapper(Mapper):
    @classmethod
    def forward(cls, obj: Optional[MediaElement], **options) -> Dict[str, Any]:
        if obj is None:
            return {"type": int(MediaType.NONE)}
        if obj._type == MediaType.IMAGE:
            return ImageMapper.forward(obj, **options)
        #  if obj._type == MediaType.POINT_CLOUD:
        #      return PointCloudMapper.forward(obj)
        if obj._type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.forward(obj)
        raise DatumaroError(f"{obj._type} is not allowed for MediaMapper.")

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> Optional[MediaElement]:
        (media_type,) = struct.unpack_from("<I", obj["type"], 0)

        if media_type == MediaType.NONE:
            return None
        if media_type == MediaType.IMAGE:
            return ImageMapper.backward(obj)
        #  if media_type == MediaType.POINT_CLOUD:
        #      return PointCloudMapper.backward(obj)
        if media_type == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward(obj)
        raise DatumaroError(f"{media_type} is not allowed for MediaMapper.")

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[Optional[MediaElement]]:
        types = pa_batches_decoder(batches, f"{parent}.type" if parent else "type")
        assert len(set(types)) == 1, "The types in batch are not identical."

        if types[0] == MediaType.NONE:
            return [None for _ in types]
        if types[0] == MediaType.IMAGE:
            return ImageMapper.backward_from_batches(batches, parent)
        if types[0] == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward_from_batches(batches, parent)
        raise NotImplementedError


class MediaElementMapper(Mapper):
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> Dict[str, Any]:
        return {
            "type": int(obj.type),
            "path": obj.path,
        }

    @classmethod
    def backward_dict(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        obj = obj.copy()
        media_type = obj["type"]
        assert media_type == cls.MEDIA_TYPE, f"Expect {cls.MEDIA_TYPE} but actual is {media_type}."
        obj["type"] = media_type
        return obj

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> MediaElement:
        media_dict = cls.backward_dict(obj)
        return MediaElement(path=media_dict["path"])

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[MediaElement]:
        paths = pa_batches_decoder(batches, f"{parent}.path" if parent else "path")
        return [MediaElement(path=path) for path in paths]


class ImageMapper(MediaElementMapper):
    MEDIA_TYPE = MediaType.IMAGE

    @classmethod
    def forward(
        cls, obj: Image, encoder: Union[str, Callable[[str, np.ndarray], bytes]] = "JPEG/75"
    ) -> Dict[str, Any]:
        out = super().forward(obj)

        options = {"path": None, "data": None}
        if os.path.exists(out["path"]):
            options["path"] = out["path"]
        else:
            options["data"] = obj.data

        if options.get("data", "NO_DATA") is not None:
            if isinstance(encoder, Callable):
                _bytes = encoder(**options)
            else:
                _bytes = ImageFileMapper.forward(**options, scheme=encoder)
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

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[Image]:
        paths = pa_batches_decoder(batches, f"{parent}.path" if parent else "path")
        attributes_ = pa_batches_decoder(batches, f"{parent}.attributes" if parent else "attributes")
        attributes_ = [DictMapper.backward(attributes)[0] for attributes in attributes_]

        images = []
        def image_decoder(path, idx):
            options = {
                "path": path if os.path.exists(path) else None,
                "data": pa_batches_decoder(batches, f"{parent}.bytes" if parent else "bytes")[idx],
            }
            return ImageFileMapper.backward(**options)

        for idx, (path, attributes) in enumerate(zip(paths, attributes_)):
            images.append(Image(data=partial(image_decoder, idx=idx), path=path, size=attributes["size"]))
        return images
