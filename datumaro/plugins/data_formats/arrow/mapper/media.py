# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


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
    def backward(cls, obj: Dict[str, Any]) -> Optional[MediaElement]:
        media_type = obj["type"]

        if media_type == MediaType.NONE:
            return None
        if media_type == MediaType.IMAGE:
            return ImageMapper.backward(obj)
        if media_type == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward(obj)
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
        if types[0] == MediaType.POINT_CLOUD:
            return PointCloudMapper.backward_from_batches(batches, parent)
        if types[0] == MediaType.MEDIA_ELEMENT:
            return MediaElementMapper.backward_from_batches(batches, parent)
        raise NotImplementedError


class MediaElementMapper(Mapper):
    MAGIC_PATH = "/NOT/A/REAL/PATH"
    MEDIA_TYPE = MediaType.MEDIA_ELEMENT

    @classmethod
    def forward(cls, obj: MediaElement) -> Dict[str, Any]:
        return {
            "type": int(obj.type),
            "path": getattr(obj, "path", cls.MAGIC_PATH),
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
        _ = cls.backward_dict(obj)
        return MediaElement()

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[MediaElement]:
        types = pa_batches_decoder(batches, f"{parent}.type" if parent else "type")
        return [MediaElement() for _ in types]


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

        out["attributes"] = DictMapper.forward(dict(size=obj.size))

        return out

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> Image:
        media_dict = cls.backward_dict(obj)

        path = media_dict["path"]
        attributes, _ = DictMapper.backward(media_dict["attributes"], 0)

        _bytes = None
        if media_dict.get("bytes", None) is not None:
            _bytes = media_dict["bytes"]

        if _bytes:
            return Image.from_bytes(data=_bytes, size=attributes["size"])
        return Image.from_file(path=path, size=attributes["size"])

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[Image]:
        paths = pa_batches_decoder(batches, f"{parent}.path" if parent else "path")
        attributes_ = pa_batches_decoder(
            batches, f"{parent}.attributes" if parent else "attributes"
        )
        attributes_ = [DictMapper.backward(attributes)[0] for attributes in attributes_]

        images = []

        def data_loader(path, idx):
            options = {
                "path": path if os.path.exists(path) else None,
                "data": pa_batches_decoder(batches, f"{parent}.bytes" if parent else "bytes")[idx],
            }
            return cls.decode(**options)

        for idx, (path, attributes) in enumerate(zip(paths, attributes_)):
            if os.path.exists(path):
                images.append(Image.from_file(path=path, size=attributes["size"]))
            else:
                images.append(
                    Image.from_bytes(
                        data=partial(data_loader, idx=idx, path=path), size=attributes["size"]
                    )
                )
        return images


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
        else:
            _bytes = obj.data
        out["bytes"] = _bytes

        bytes_arr = bytearray()
        bytes_arr.extend(struct.pack("<I", len(obj.extra_images)))
        for img in obj.extra_images:
            bytes_arr.extend(
                DictMapper.forward(
                    b64encode(ImageMapper.forward(img, encoder=encoder), cls.B64_PREFIX)
                )
            )
        out["attributes"] = bytes(bytes_arr)

        return out

    @classmethod
    def backward(cls, obj: Dict[str, Any]) -> PointCloud:
        offset = 0
        media_dict = cls.backward_dict(obj)

        path = media_dict["path"]
        (len_extra_images,) = struct.unpack_from("<I", media_dict["attributes"], offset)
        offset += 4

        _bytes = None
        if media_dict.get("bytes", None) is not None:
            _bytes = media_dict["bytes"]

        extra_images = []
        for _ in range(len_extra_images):
            img, offset = DictMapper.backward(media_dict["attributes"], offset)
            extra_images.append(ImageMapper.backward(b64decode(img, cls.B64_PREFIX)))

        if _bytes:
            return PointCloud.from_bytes(data=_bytes, extra_images=extra_images)
        else:
            return PointCloud.from_file(path=path, extra_images=extra_images)

    @classmethod
    def backward_from_batches(
        cls,
        batches: List[pa.lib.RecordBatch],
        parent: Optional[str] = None,
    ) -> List[PointCloud]:
        paths = pa_batches_decoder(batches, f"{parent}.path" if parent else "path")

        def data_loader(idx):
            data = pa_batches_decoder(batches, f"{parent}.bytes" if parent else "bytes")[idx]
            return data

        def extra_images(idx):
            offset = 0
            attributes = pa_batches_decoder(
                batches, f"{parent}.attributes" if parent else "attributes"
            )[idx]
            (len_extra_images,) = struct.unpack_from("<I", attributes, offset)
            offset += 4
            outs = []
            for _ in range(len_extra_images):
                img, offset = DictMapper.backward(attributes, offset)
                outs.append(ImageMapper.backward(b64decode(img, cls.B64_PREFIX)))
            return outs

        point_clouds = []
        for idx, path in enumerate(paths):
            if os.path.exists(path):
                point_clouds.append(
                    PointCloud.from_file(
                        path=path,
                        extra_images=partial(extra_images, idx=idx),
                    )
                )
            else:
                point_clouds.append(
                    PointCloud.from_bytes(
                        data=partial(data_loader, idx=idx),
                        extra_images=partial(extra_images, idx=idx),
                    )
                )
        return point_clouds
