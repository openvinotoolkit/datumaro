# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import importlib
import os
import os.path as osp
import shlex
import warnings
import weakref
from enum import Enum, auto
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

try:
    # Introduced in 1.20
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any


class _IMAGE_BACKENDS(Enum):
    cv2 = auto()
    PIL = auto()


_IMAGE_BACKEND = None
_image_loading_errors = (FileNotFoundError,)
try:
    importlib.import_module("cv2")
    _IMAGE_BACKEND = _IMAGE_BACKENDS.cv2
except ModuleNotFoundError:
    import PIL

    _IMAGE_BACKEND = _IMAGE_BACKENDS.PIL
    _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

from datumaro.util.image_cache import ImageCache
from datumaro.util.os_util import find_files


def __getattr__(name: str):
    if name in {"Image", "ByteImage"}:
        warnings.warn(
            f"Using {name} from 'util.image' is deprecated, "
            "the class is moved to 'components.media'",
            DeprecationWarning,
            stacklevel=2,
        )

        import datumaro.components.media as media_module

        return getattr(media_module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def load_image(path: str, dtype: DTypeLike = np.float32):
    """
    Reads an image in the HWC Grayscale/BGR(A) float [0; 255] format.
    """

    if _IMAGE_BACKEND == _IMAGE_BACKENDS.cv2:
        # cv2.imread does not support paths that are not representable
        # in the locale encoding on Windows, so we read the image bytes
        # ourselves.

        with open(path, "rb") as f:
            image_bytes = f.read()

        return decode_image(image_bytes, dtype=dtype)
    elif _IMAGE_BACKEND == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        image = Image.open(path)
        image = np.asarray(image, dtype=dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # RGB to BGR
    else:
        raise NotImplementedError()

    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image


def save_image(
    path: str, image: np.ndarray, create_dir: bool = False, dtype: DTypeLike = np.uint8, **kwargs
) -> None:
    # NOTE: Check destination path for existence
    # OpenCV silently fails if target directory does not exist
    dst_dir = osp.dirname(path)
    if dst_dir:
        if create_dir:
            os.makedirs(dst_dir, exist_ok=True)
        elif not osp.isdir(dst_dir):
            raise FileNotFoundError("Directory does not exist: '%s'" % dst_dir)

    if not kwargs:
        kwargs = {}

    # NOTE: OpenCV documentation says "If the image format is not supported,
    # the image will be converted to 8-bit unsigned and saved that way".
    # Conversion from np.int32 to np.uint8 is not working properly
    backend = _IMAGE_BACKEND
    if dtype == np.int32:
        backend = _IMAGE_BACKENDS.PIL
    if backend == _IMAGE_BACKENDS.cv2:
        # cv2.imwrite does not support paths that are not representable
        # in the locale encoding on Windows, so we write the image bytes
        # ourselves.

        ext = osp.splitext(path)[1]
        image_bytes = encode_image(image, ext, dtype=dtype, **kwargs)

        with open(path, "wb") as f:
            f.write(image_bytes)
    elif backend == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        params = {}
        params["quality"] = kwargs.get("jpeg_quality")
        if kwargs.get("jpeg_quality") == 100:
            params["subsampling"] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # BGR to RGB
        image = Image.fromarray(image)
        image.save(path, **params)
    else:
        raise NotImplementedError()


def encode_image(image: np.ndarray, ext: str, dtype: DTypeLike = np.uint8, **kwargs) -> bytes:
    if not kwargs:
        kwargs = {}

    if _IMAGE_BACKEND == _IMAGE_BACKENDS.cv2:
        import cv2

        params = []

        if not ext.startswith("."):
            ext = "." + ext

        if ext.upper() == ".JPG":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), kwargs.get("jpeg_quality", 75)]

        image = image.astype(dtype)
        success, result = cv2.imencode(ext, image, params=params)
        if not success:
            raise Exception("Failed to encode image to '%s' format" % (ext))
        return result.tobytes()
    elif _IMAGE_BACKEND == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        if ext.startswith("."):
            ext = ext[1:]

        params = {}
        params["quality"] = kwargs.get("jpeg_quality")
        if kwargs.get("jpeg_quality") == 100:
            params["subsampling"] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # BGR to RGB
        image = Image.fromarray(image)
        with BytesIO() as buffer:
            image.save(buffer, format=ext, **params)
            return buffer.getvalue()
    else:
        raise NotImplementedError()


def decode_image(image_bytes: bytes, dtype: DTypeLike = np.float32) -> np.ndarray:
    if _IMAGE_BACKEND == _IMAGE_BACKENDS.cv2:
        import cv2

        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = image.astype(dtype)
    elif _IMAGE_BACKEND == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes))
        image = np.asarray(image, dtype=dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # RGB to BGR
    else:
        raise NotImplementedError()

    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jp2",
    ".png",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".tga",
    ".webp",
    ".pfm",
    ".sr",
    ".ras",
    ".exr",
    ".hdr",
    ".pic",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
}


def find_images(
    dirpath: str,
    exts: Union[str, Iterable[str]] = None,
    recursive: bool = False,
    max_depth: Optional[int] = None,
    min_depth: Optional[int] = None,
) -> Iterator[str]:
    yield from find_files(
        dirpath,
        exts=exts or IMAGE_EXTENSIONS,
        recursive=recursive,
        max_depth=max_depth,
        min_depth=min_depth,
    )


def is_image(path: str) -> bool:
    trunk, ext = osp.splitext(osp.basename(path))
    return trunk and ext.lower() in IMAGE_EXTENSIONS and osp.isfile(path)


class lazy_image:
    def __init__(
        self,
        path: str,
        loader: Callable[[str], np.ndarray] = None,
        cache: Union[bool, ImageCache] = True,
    ) -> None:
        """
        Cache:
            - False: do not cache
            - True: use the global cache
            - ImageCache instance: an object to be used as cache
        """

        if loader is None:
            loader = load_image
        self._path = path
        self._loader = loader

        assert isinstance(cache, (ImageCache, bool))
        self._cache = cache

    def __call__(self) -> np.ndarray:
        image = None
        cache_key = weakref.ref(self)

        cache = self._get_cache()
        if cache is not None:
            image = cache.get(cache_key)

        if image is None:
            image = self._loader(self._path)
            if cache is not None:
                cache.push(cache_key, image)
        return image

    def _get_cache(self) -> Optional[ImageCache]:
        if self._cache is True:
            cache = ImageCache.get_instance()
        elif self._cache is False:
            cache = None
        else:
            cache = self._cache
        return cache


ImageMeta = Dict[str, Tuple[int, int]]
"""filename -> height, width"""

DEFAULT_IMAGE_META_FILE_NAME = "images.meta"


def load_image_meta_file(image_meta_path: str) -> ImageMeta:
    """
    Loads image metadata from a file with the following format:

        <image name 1> <height 1> <width 1>

        <image name 2> <height 2> <width 2>

        ...

    Shell-like comments and quoted fields are allowed.

    This can be useful to support datasets in which image dimensions are
    required to interpret annotations.
    """
    assert isinstance(image_meta_path, str)

    if not osp.isfile(image_meta_path):
        raise FileNotFoundError("Can't read image meta file '%s'" % image_meta_path)

    image_meta = {}

    with open(image_meta_path, encoding="utf-8") as f:
        for line in f:
            fields = shlex.split(line, comments=True)
            if not fields:
                continue

            # ignore extra fields, so that the format can be extended later
            image_name, h, w = fields[:3]
            image_meta[image_name] = (int(h), int(w))

    return image_meta


def save_image_meta_file(image_meta: ImageMeta, image_meta_path: str) -> None:
    """
    Saves image_meta to the path specified by image_meta_path in the format
    defined in load_image_meta_file's documentation.
    """

    assert isinstance(image_meta_path, str)

    with open(image_meta_path, "w", encoding="utf-8") as f:
        # Add a comment about file syntax
        print("# <image name> <height> <width>", file=f)
        print("", file=f)

        for image_name, (height, width) in image_meta.items():
            print(shlex.quote(image_name), height, width, file=f)
