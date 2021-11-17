# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable, Iterable, Iterator, Optional, Tuple, Union
import os
import os.path as osp
import shutil

import cv2
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
        """Media file extension"""
        return osp.splitext(osp.basename(self.path))[1]

    def __eq__(self, other: object) -> bool:
        # We need to compare exactly with this type
        if type(other) is not __class__: # pylint: disable=unidiomatic-typecheck
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
            return False
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

class VideoFrame(Image):
    def __init__(self, video: 'Video', index: int):
        self._video = video
        self._index = index

        super().__init__(lambda _: self._video.get_frame_data(self._index))

    @property
    def size(self) -> Tuple[int, int]:
        return self._video.frame_size

    @property
    def index(self) -> int:
        return self._index

    @property
    def video(self) -> 'Video':
        return self._video

class _VideoFrameIterator(Iterator[VideoFrame]):
    """
    Provides sequential access to the video frames.
    """

    def __init__(self, video: 'Video'):
        self._video: 'Video' = video
        self._iterator: Iterator[VideoFrame] = None
        self._pos: int = -1
        self._current_frame_data: Optional[np.ndarray] = None

        self._reset()

    def _reset(self):
        self._video._reset()
        self._iterator = self._decode(self._video._get_container())
        self._pos = -1
        self._current_frame_data = None

    def _decode(self, cap) -> Iterator[VideoFrame]:
        """
        Decodes video frames using opencv
        """

        self._pos = -1

        success, frame = cap.read()
        while success:
            self._pos += 1
            if self._video._includes_frame(self._pos):
                self._current_frame_data = frame.astype(float)
                yield VideoFrame(self, index=self._pos)

            success, frame = cap.read()

        if self._video._frame_count is None:
            self._video._frame_count = self._pos + 1

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def __getitem__(self, idx: int) -> VideoFrame:
        if not self._video._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._navigate_to(idx)

    def get_frame_data(self, idx: int) -> np.ndarray:
        self._navigate_to(idx)
        return self._current_frame_data

    def _navigate_to(self, idx: int) -> VideoFrame:
        """
        Iterates over frames to the required position.
        """

        if idx < 0:
            raise IndexError()

        if idx < self._pos:
            self._reset()

        if self._pos < idx:
            try:
                while self._pos < idx:
                    v = self.__next__()
            except StopIteration as e:
                raise IndexError() from e
        else:
            v = VideoFrame(video=self, index=self._pos)

        return v

class Video(MediaElement, Iterable[VideoFrame]):
    """
    Provides random access to the video frames.
    """

    def __init__(self, path: str, *,
            step: int = 1, start_frame: int = 0,
            end_frame: Optional[int] = None) -> None:
        super().__init__(path)

        self._iterator = None
        self._frame_size: Optional[Tuple[int, int]] = None

        self._container = None
        self._reset()

        self._step = int(step)
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame) if end_frame else None

        self._frame_count = self._get_frame_count(self._container)
        self._length = None
        self._get_length()

    def close(self):
        self._iterator = None

        if self._container is not None:
            self._container.release()
            self._container = None

    def __getitem__(self, idx: int) -> VideoFrame:
        if not self._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._get_iterator()[idx]

    def get_frame_data(self, idx: int) -> VideoFrame:
        if not self._includes_frame(idx):
            raise IndexError(f"Video doesn't contain frame #{idx}.")

        return self._get_iterator().get_frame_data(idx)

    def __iter__(self) -> Iterator[VideoFrame]:
        """
        Iterates over frames lazily, if possible.
        """

        if self._frame_count is not None:
            # Decoding is not necessary to get frame pointers
            # However, it can be inacurrate
            end_frame = self._get_end_frame()
            for index in range(self._start_frame, end_frame, self._step):
                yield VideoFrame(video=self, index=index)
        else:
            # Need to decode to iterate over frames
            yield from self._get_iterator()

    @property
    def frame_count(self) -> Optional[int]:
        """
        Returns frame count, if video provides such information.
        Note that not all videos provide length / duration metainfo.

        The count is affected by the frame filtering options of the object.
        """
        return self._get_length()

    @property
    def frame_size(self) -> Tuple[int, int]:
        if self._frame_size is None:
            self._frame_size = self._get_frame_size()
        return self._frame_size

    def _get_frame_size(self) -> Tuple[int, int]:
        w = self._container.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self._container.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if h and w:
            frame_size = (int(h), int(w))
        else:
            image = next(self._get_iterator()).data
            frame_size = image.shape[0:2]

        return frame_size

    @staticmethod
    def _get_frame_count(container) -> Optional[int]:
        # Not all videos provide length / duration metainfo
        # Note that this information can be invalid or inaccurate
        # due to variable frame rate
        # https://stackoverflow.com/a/47796468
        video_length = container.get(cv2.CAP_PROP_FRAME_COUNT)
        if video_length:
            video_length = int(video_length)
        else:
            video_length = None
        return video_length

    def _get_end_frame(self):
        if self._end_frame is not None and self._frame_count is not None:
            end_frame = min(self._end_frame, self._frame_count)
        else:
            end_frame = self._end_frame or self._frame_count

        return end_frame

    def _get_length(self):
        if self._length is None:
            end_frame = self._get_end_frame()

            length = None
            if end_frame is not None:
                length = (end_frame - self._start_frame) // self._step
                assert 0 < length

            self._length = length

        return self._length

    def _includes_frame(self, i):
        end_frame = self._get_end_frame()
        if self._start_frame <= i:
            if (i - self._start_frame) % self._step == 0:
                if end_frame is None or i < end_frame:
                    return True

        return False

    def _get_iterator(self):
        if self._iterator is None:
            self._iterator = _VideoFrameIterator(self)
        return self._iterator

    def _get_container(self):
        if self._container is None:
            self._reset()
        return self._container

    def _open(self):
        return cv2.VideoCapture(self._path)

    def _reset(self):
        if self._container is not None:
            self._container.release()
        self._container = self._open()
        assert self._container.isOpened()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, __class__):
            return False

        return self.path == other.path and \
            self._start_frame == other._start_frame and \
            self._step == other._step and \
            self._end_frame == other._end_frame
