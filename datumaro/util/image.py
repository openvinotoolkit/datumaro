# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from io import BytesIO
from typing import (
    Any, Callable, Dict, Iterable, Iterator, Optional, Tuple, Union,
)
import importlib
import os
import os.path as osp
import shlex
import shutil
import weakref

import numpy as np


class _IMAGE_BACKENDS(Enum):
    cv2 = auto()
    PIL = auto()
_IMAGE_BACKEND = None
_image_loading_errors = (FileNotFoundError, )
try:
    importlib.import_module('cv2')
    _IMAGE_BACKEND = _IMAGE_BACKENDS.cv2
except ImportError:
    import PIL
    _IMAGE_BACKEND = _IMAGE_BACKENDS.PIL
    _image_loading_errors = (*_image_loading_errors, PIL.UnidentifiedImageError)

from datumaro.util.image_cache import ImageCache as _ImageCache
from datumaro.util.os_util import walk


def load_image(path, dtype=np.float32):
    """
    Reads an image in the HWC Grayscale/BGR(A) float [0; 255] format.
    """

    if _IMAGE_BACKEND == _IMAGE_BACKENDS.cv2:
        # cv2.imread does not support paths that are not representable
        # in the locale encoding on Windows, so we read the image bytes
        # ourselves.

        with open(path, 'rb') as f:
            image_bytes = f.read()

        return decode_image(image_bytes, dtype=dtype)
    elif _IMAGE_BACKEND == _IMAGE_BACKENDS.PIL:
        from PIL import Image
        image = Image.open(path)
        image = np.asarray(image, dtype=dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1] # RGB to BGR
    else:
        raise NotImplementedError()

    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image

def save_image(path, image, create_dir=False, dtype=np.uint8, **kwargs):
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

        with open(path, 'wb') as f:
            f.write(image_bytes)
    elif backend == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        params = {}
        params['quality'] = kwargs.get('jpeg_quality')
        if kwargs.get('jpeg_quality') == 100:
            params['subsampling'] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1] # BGR to RGB
        image = Image.fromarray(image)
        image.save(path, **params)
    else:
        raise NotImplementedError()

def encode_image(image, ext, dtype=np.uint8, **kwargs):
    if not kwargs:
        kwargs = {}

    if _IMAGE_BACKEND == _IMAGE_BACKENDS.cv2:
        import cv2

        params = []

        if not ext.startswith('.'):
            ext = '.' + ext

        if ext.upper() == '.JPG':
            params = [
                int(cv2.IMWRITE_JPEG_QUALITY), kwargs.get('jpeg_quality', 75)
            ]

        image = image.astype(dtype)
        success, result = cv2.imencode(ext, image, params=params)
        if not success:
            raise Exception("Failed to encode image to '%s' format" % (ext))
        return result.tobytes()
    elif _IMAGE_BACKEND == _IMAGE_BACKENDS.PIL:
        from PIL import Image

        if ext.startswith('.'):
            ext = ext[1:]

        params = {}
        params['quality'] = kwargs.get('jpeg_quality')
        if kwargs.get('jpeg_quality') == 100:
            params['subsampling'] = 0

        image = image.astype(dtype)
        if len(image.shape) == 3 and image.shape[2] in {3, 4}:
            image[:, :, :3] = image[:, :, 2::-1] # BGR to RGB
        image = Image.fromarray(image)
        with BytesIO() as buffer:
            image.save(buffer, format=ext, **params)
            return buffer.getvalue()
    else:
        raise NotImplementedError()

def decode_image(image_bytes, dtype=np.float32):
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
            image[:, :, :3] = image[:, :, 2::-1] # RGB to BGR
    else:
        raise NotImplementedError()

    assert len(image.shape) in {2, 3}
    if len(image.shape) == 3:
        assert image.shape[2] in {3, 4}
    return image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.jpe', '.jp2',
    '.png', '.bmp', '.dib', '.tif', '.tiff', '.tga', '.webp', '.pfm',
    '.sr', '.ras', '.exr', '.hdr', '.pic',
    '.pbm', '.pgm', '.ppm', '.pxm', '.pnm',
}

def find_images(dirpath: str, exts: Union[str, Iterable[str]] = None,
        recursive: bool = False, max_depth: int = None) -> Iterator[str]:
    if isinstance(exts, str):
        exts = ['.' + exts.lower().lstrip('.')]
    elif exts is None:
        exts = IMAGE_EXTENSIONS
    else:
        exts = list('.' + e.lower().lstrip('.') for e in exts)

    def _check_image_ext(filename: str):
        dotpos = filename.rfind('.')
        if 0 < dotpos: # exclude '.ext' cases too
            ext = filename[dotpos:].lower()
            if ext in exts:
                return True
        return False

    for d, _, filenames in walk(dirpath,
            max_depth=max_depth if recursive else 0):
        for filename in filenames:
            if not _check_image_ext(filename):
                continue

            yield osp.join(d, filename)


class lazy_image:
    def __init__(self, path, loader=None, cache=None):
        if loader is None:
            loader = load_image
        self._path = path
        self._loader = loader

        # Cache:
        # - False: do not cache
        # - None: use the global cache
        # - object: an object to be used as cache
        assert cache in {None, False} or isinstance(cache, object)
        self._cache = cache

    def __call__(self):
        image = None
        cache_key = weakref.ref(self)

        cache = self._get_cache(self._cache)
        if cache is not None:
            image = cache.get(cache_key)

        if image is None:
            image = self._loader(self._path)
            if cache is not None:
                cache.push(cache_key, image)
        return image

    @staticmethod
    def _get_cache(cache):
        if cache is None:
            cache = _ImageCache.get_instance()
        elif cache == False:
            return None
        return cache

class Image:
    def __init__(self, data: Union[None, Callable, np.ndarray] = None,
            path: Optional[str] = None, loader: Optional[Callable] = None,
            size: Optional[Tuple[int, int]] = None, cache: Any = None):
        assert size is None or len(size) == 2, size
        if size is not None:
            assert len(size) == 2 and 0 < size[0] and 0 < size[1], size
            size = tuple(map(int, size))
        self._size = size # (H, W)

        assert path is None or isinstance(path, str), path
        if path is None:
            path = ''
        elif path:
            path = osp.abspath(path).replace('\\', '/')
        self._path = path

        assert data is not None or path or loader, "Image can not be empty"
        if data is not None:
            assert callable(data) or isinstance(data, np.ndarray), type(data)
        if data is None and (path or loader):
            if osp.isfile(path) or loader:
                data = lazy_image(path, loader=loader, cache=cache)
        self._data = data

        if not self._size and isinstance(data, np.ndarray):
            self._size = data.shape[:2]

    @property
    def path(self) -> str:
        return self._path

    @property
    def ext(self) -> str:
        return osp.splitext(osp.basename(self.path))[1]

    @property
    def data(self) -> np.ndarray:
        if callable(self._data):
            data = self._data()
        else:
            data = self._data

        if self._size is None and data is not None:
            self._size =  tuple(map(int, data.shape[:2]))
        return data

    @property
    def has_data(self) -> bool:
        return self._data is not None

    @property
    def has_size(self) -> bool:
        return self._size is not None or isinstance(self._data, np.ndarray)

    @property
    def size(self) -> Optional[Tuple[int, int]]:
        if self._size is None:
            try:
                data = self.data
            except _image_loading_errors:
                return None
            if data is not None:
                self._size = tuple(map(int, data.shape[:2]))
        return self._size

    def __eq__(self, other):
        if isinstance(other, np.ndarray):
            return self.has_data and np.array_equal(self.data, other)

        if not isinstance(other, __class__):
            return False
        return \
            (np.array_equal(self.size, other.size)) and \
            (self.has_data == other.has_data) and \
            (self.has_data and np.array_equal(self.data, other.data) or \
                not self.has_data)

    def save(self, path):
        src_ext = self.ext.lower()
        dst_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if src_ext == dst_ext and osp.isfile(self.path):
            if self.path != path:
                shutil.copyfile(self.path, path)
        else:
            save_image(path, self.data)

class ByteImage(Image):
    def __init__(self, data=None, path=None, ext=None, cache=None, size=None):
        loader = None
        if data is not None:
            if callable(data) and not isinstance(data, lazy_image):
                data = lazy_image(path, loader=data, cache=cache)
            loader = lambda _: decode_image(self.get_bytes())

        super().__init__(path=path, size=size, loader=loader, cache=cache)
        if data is None and loader is None:
            # unset defaults for regular images
            # to avoid random file reading to bytes
            self._data = None

        self._bytes_data = data
        if ext:
            ext = ext.lower()
            if not ext.startswith('.'):
                ext = '.' + ext
        self._ext = ext

    def get_bytes(self):
        if callable(self._bytes_data):
            return self._bytes_data()
        return self._bytes_data

    @property
    def ext(self):
        if self._ext:
            return self._ext
        return super().ext

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return super().__eq__(other)
        return \
            (np.array_equal(self.size, other.size)) and \
            (self.has_data == other.has_data) and \
            (self.has_data and self.get_bytes() == other.get_bytes() or \
                not self.has_data)

    def save(self, path):
        src_ext = self.ext.lower()
        dst_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if src_ext == dst_ext and osp.isfile(self.path):
            if self.path != path:
                shutil.copyfile(self.path, path)
        elif src_ext == dst_ext:
            with open(path, 'wb') as f:
                f.write(self.get_bytes())
        else:
            save_image(path, self.data)

ImageMeta = Dict[str, Tuple[int, int]]

DEFAULT_IMAGE_META_FILE_NAME = 'images.meta'

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
        raise Exception("Can't read image meta file '%s'" % image_meta_path)

    image_meta = {}

    with open(image_meta_path, encoding='utf-8') as f:
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

    with open(image_meta_path, 'w', encoding='utf-8') as f:
        for image_name, (height, width) in image_meta.items():
            print(shlex.quote(image_name), height, width, file=f)
