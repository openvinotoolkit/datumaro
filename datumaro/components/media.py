# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable, Optional, Tuple, Union
import os
import os.path as osp
import shutil

import numpy as np

from datumaro.util.image import (
    _image_loading_errors, decode_image, lazy_image, save_image,
)


class MediaElement:
    def __init__(self, path: str) -> None:
        self._path = path

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Image file extension"""
        return osp.splitext(osp.basename(self.path))[1]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, __class__):
            return False
        return self._path == other._path

class Image(MediaElement):
    def __init__(self,
            data: Union[np.ndarray, Callable[[str], np.ndarray], None] = None,
            *,
            path: Optional[str] = None,
            size: Optional[Tuple[int, int]] = None):
        assert size is None or len(size) == 2, size
        if size is not None:
            assert len(size) == 2 and 0 < size[0] and 0 < size[1], size
            size = tuple(map(int, size))
        self._size = size # (H, W)
        if not self._size and isinstance(data, np.ndarray):
            self._size = data.shape[:2]

        assert path is None or isinstance(path, str), path
        if path is None:
            path = ''
        elif path:
            path = osp.abspath(path).replace('\\', '/')
        self._path = path

        if not isinstance(data, np.ndarray):
            assert path or callable(data), "Image can not be empty"
            assert data is None or callable(data)
            if path and osp.isfile(path) or data:
                data = lazy_image(path, loader=data)
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """Image data in BGR HWC [0; 255] (float) format"""

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
        """Returns (H, W)"""

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
            return super().__eq__(other)
        return \
            (np.array_equal(self.size, other.size)) and \
            (self.has_data == other.has_data) and \
            (self.has_data and np.array_equal(self.data, other.data) or \
                not self.has_data)

    def save(self, path):
        cur_path = osp.abspath(self.path)
        path = osp.abspath(path)

        cur_ext = self.ext.lower()
        new_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if cur_ext == new_ext and osp.isfile(cur_path):
            if cur_path != path:
                shutil.copyfile(cur_path, path)
        else:
            save_image(path, self.data)

class ByteImage(Image):
    def __init__(self,
            data: Union[bytes, Callable[[str], bytes], None] = None,
            *,
            path: Optional[str] = None,
            ext: Optional[str] = None,
            size: Optional[Tuple[int, int]] = None):
        if not isinstance(data, bytes):
            assert path or callable(data), "Image can not be empty"
            assert data is None or callable(data)
            if path and osp.isfile(path) or data:
                data = lazy_image(path, loader=data)

        super().__init__(path=path, size=size,
            data=lambda _: decode_image(self.get_bytes()))
        if data is None:
            # We don't expect decoder to produce images from nothing,
            # otherwise using this class makes no sense. We undefine
            # data to avoid using default image loader for loading binaries
            # from the path, when no data is provided.
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
        cur_path = osp.abspath(self.path)
        path = osp.abspath(path)

        cur_ext = self.ext.lower()
        new_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if cur_ext == new_ext and osp.isfile(cur_path):
            if cur_path != path:
                shutil.copyfile(cur_path, path)
        elif cur_ext == new_ext:
            with open(path, 'wb') as f:
                f.write(self.get_bytes())
        else:
            save_image(path, self.data)
