# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib
import os
import os.path as osp
import shlex
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from io import BytesIO, IOBase
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np

from datumaro.components.crypter import NULL_CRYPTER, Crypter
from datumaro.components.errors import DatumaroError

try:
    # Introduced in 1.20
    from numpy.typing import DTypeLike
except ImportError:
    DTypeLike = Any


class _IMAGE_BACKENDS(Enum):
    cv2 = auto()
    PIL = auto()


_IMAGE_BACKEND: ContextVar[_IMAGE_BACKENDS] = ContextVar("_IMAGE_BACKENDS")
_image_loading_errors = (FileNotFoundError,)
try:
    importlib.import_module("cv2")
    _IMAGE_BACKEND.set(_IMAGE_BACKENDS.cv2)
except ModuleNotFoundError:
    import PIL

    _IMAGE_BACKEND.set(_IMAGE_BACKENDS.PIL)
    _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

from datumaro.util.image_cache import ImageCache
from datumaro.util.os_util import find_files

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class ImageColorScale(Enum):
    """Image color scale

    - UNCHANGED: Use the original image's scale (default)
    - COLOR: Use 3 channels (it can ignore the alpha channel or convert the gray scale image to BGR)
    """

    UNCHANGED = 0
    COLOR = 1

    def convert_cv2(self, img: np.ndarray) -> np.ndarray:
        import cv2

        if self == ImageColorScale.COLOR:
            return cv2.imdecode(img, cv2.IMREAD_COLOR)

        return cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

    def convert_pil(self, img: PILImage) -> PILImage:
        if self == ImageColorScale.COLOR:
            return img.convert("RGB")

        return img


IMAGE_COLOR_SCALE: ContextVar[ImageColorScale] = ContextVar(
    "IMAGE_COLOR_SCALE", default=ImageColorScale.UNCHANGED
)


@contextmanager
def decode_image_context(color_scale: ImageColorScale):
    """Change Datumaro image decoding color scale.

    For model training, it is recommended to use this context manager
    to load images in the BGR 3-channel format. For example,

    .. code-block:: python

        import datumaro as dm
        with decode_image_context(ImageColorScale.COLOR):
            item: dm.DatasetItem
            img_data = item.media_as(dm.Image).data
            assert img_data.shape[-1] == 3  # It should be a 3-channel image
    """
    curr_ctx = IMAGE_COLOR_SCALE.get()
    IMAGE_COLOR_SCALE.set(color_scale)
    yield
    IMAGE_COLOR_SCALE.set(curr_ctx)


def load_image(path: str, dtype: DTypeLike = np.uint8, crypter: Crypter = NULL_CRYPTER):
    """
    Reads an image in the HWC Grayscale/BGR(A) [0; 255] format (default dtype is uint8).
    """
    _image_loading_errors = (FileNotFoundError,)
    try:
        importlib.import_module("cv2")
        _IMAGE_BACKEND.set(_IMAGE_BACKENDS.cv2)
    except ModuleNotFoundError:
        import PIL

        _IMAGE_BACKEND.set(_IMAGE_BACKENDS.PIL)
        _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

    _image_loading_errors = (FileNotFoundError,)
    try:
        importlib.import_module("cv2")
        _IMAGE_BACKEND.set(_IMAGE_BACKENDS.cv2)
    except ModuleNotFoundError:
        import PIL

        _IMAGE_BACKEND.set(_IMAGE_BACKENDS.PIL)
        _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

    if _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.cv2:
        # cv2.imread does not support paths that are not representable
        # in the locale encoding on Windows, so we read the image bytes
        # ourselves.

        with open(path, "rb") as f:
            image_bytes = crypter.decrypt(f.read())

        return decode_image(image_bytes, dtype=dtype)
    elif _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.PIL:
        with open(path, "rb") as f:
            image_bytes = crypter.decrypt(f.read())

        return decode_image(image_bytes, dtype=dtype)

    raise NotImplementedError(_IMAGE_BACKEND)


def copyto_image(
    src: Union[str, IOBase], dst: Union[str, IOBase], src_crypter: Crypter, dst_crypter: Crypter
) -> None:
    if src_crypter == dst_crypter and src == dst:
        return

    @contextmanager
    def _open(fp, mode):
        was_file = False
        if not isinstance(fp, IOBase):
            was_file = True
            fp = open(fp, mode)
        yield fp
        if was_file:
            fp.close()

    with _open(src, "rb") as src_fp:
        _bytes = src_crypter.decrypt(src_fp.read())

    with _open(dst, "wb") as dst_fp:
        dst_fp.write(dst_crypter.encrypt(_bytes))


def save_image(
    dst: Union[str, IOBase],
    image: np.ndarray,
    ext: Optional[str] = None,
    create_dir: bool = False,
    dtype: DTypeLike = np.uint8,
    crypter: Crypter = NULL_CRYPTER,
    **kwargs,
) -> None:
    # NOTE: Check destination path for existence
    # OpenCV silently fails if target directory does not exist
    if isinstance(dst, IOBase):
        ext = ext if ext else ".png"
    else:
        dst_dir = osp.dirname(dst)
        if dst_dir:
            if create_dir:
                os.makedirs(dst_dir, exist_ok=True)
            elif not osp.isdir(dst_dir):
                raise FileNotFoundError("Directory does not exist: '%s'" % dst_dir)
        # file extension and actual encoding can be differ
        ext = ext if ext else osp.splitext(dst)[1]

    if not kwargs:
        kwargs = {}

    # NOTE: OpenCV documentation says "If the image format is not supported,
    # the image will be converted to 8-bit unsigned and saved that way".
    # Conversion from np.int32 to np.uint8 is not working properly
    backend = _IMAGE_BACKEND.get()
    if dtype == np.int32:
        backend = _IMAGE_BACKENDS.PIL
    if backend == _IMAGE_BACKENDS.cv2:
        # cv2.imwrite does not support paths that are not representable
        # in the locale encoding on Windows, so we write the image bytes
        # ourselves.

        image_bytes = encode_image(image, ext, dtype=dtype, **kwargs)

        if isinstance(dst, str):
            with open(dst, "wb") as f:
                f.write(crypter.encrypt(image_bytes))
        else:
            dst.write(crypter.encrypt(image_bytes))
    elif backend == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        if ext.startswith("."):
            ext = ext[1:]

        if not crypter.is_null_crypter:
            raise DatumaroError("PIL backend should have crypter=NullCrypter.")

        params = {}
        params["quality"] = kwargs.get("jpeg_quality")
        if kwargs.get("jpeg_quality") == 100:
            params["subsampling"] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1]  # BGR to RGB
        image = Image.fromarray(image)
        image.save(dst, format=ext, **params)
    else:
        raise NotImplementedError()


def encode_image(image: np.ndarray, ext: str, dtype: DTypeLike = np.uint8, **kwargs) -> bytes:
    if not kwargs:
        kwargs = {}

    if _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.cv2:
        import cv2

        params = []

        if not ext.startswith("."):
            ext = "." + ext

        if ext.upper() in (".JPG", ".JPEG"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), kwargs.get("jpeg_quality", 75)]

        image = image.astype(dtype)
        success, result = cv2.imencode(ext, image, params=params)
        if not success:
            raise Exception("Failed to encode image to '%s' format" % (ext))
        return result.tobytes()
    elif _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.PIL:
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


def decode_image(image_bytes: bytes, dtype: DTypeLike = np.uint8) -> np.ndarray:
    ctx_color_scale = IMAGE_COLOR_SCALE.get()

    if _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.cv2:
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = ctx_color_scale.convert_cv2(image)
        image = image.astype(dtype)
    elif _IMAGE_BACKEND.get() == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        image = Image.open(BytesIO(image_bytes))
        image = ctx_color_scale.convert_pil(image)
        image = np.asarray(image, dtype=dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image = np.array(image)  # Release read-only
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
        crypter: Crypter = NULL_CRYPTER,
    ) -> None:
        """
        Cache:
            - False: do not cache
            - True: use the global cache
            - ImageCache instance: an object to be used as cache
        """

        self._custom_loader = True

        if loader is None:
            loader = load_image
            self._custom_loader = False

        self._path = path
        self._loader = loader

        assert isinstance(cache, (ImageCache, bool))
        self._cache = cache
        self._crypter = crypter

    def __call__(self) -> np.ndarray:
        image = None
        cache_key = weakref.ref(self)

        cache = self._get_cache()
        if cache is not None:
            image = cache.get(cache_key)

        if image is None:
            image = (
                self._loader(self._path)
                if self._custom_loader
                else self._loader(self._path, crypter=self._crypter)
            )
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
