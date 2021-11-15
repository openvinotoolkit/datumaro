# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np

from datumaro.components.image import Image
from datumaro.components.media_element import MediaElement


class VideoFrame(Image):
    def __init__(self, video: 'Video', index: int):
        self._video = video
        self._index = index

        super().__init__(self,
            data=lambda _: self._video.get_frame_data(self._index))

    @property
    def size(self) -> Tuple[int, int]:
        return self._video.get_frame_size()

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
                self._current_frame_data = frame
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

        self._frame_count = self._get_frame_count(self._container)
        self._end_frame = int(end_frame) if end_frame else self._frame_count

        self._length = None
        if self._step is not None and \
                self._start_frame is not None and \
                self._end_frame is not None:
            self._length = (self._end_frame - self._start_frame) // self._step
            assert 0 < self._length

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

        if self._frame_count is None:
            # Need to decode to iterate over frames
            return iter(self)
        else:
            # Decoding is not necessary to get frame pointers
            for index in range(self._start_frame, self._end_frame, self._step):
                yield VideoFrame(video=self, index=self._get_frame(index))

    def __len__(self) -> Optional[int]:
        return self._length

    def get_frame_size(self) -> Tuple[int, int]:
        if self._frame_size is None:
            image = next(self._get_iterator()).data
        return (image.width, image.height)

    @staticmethod
    def _get_frame_count(container) -> Optional[int]:
        # Not all videos provide length / duration metainfo
        # Note that this information can be invalid
        # https://stackoverflow.com/a/47796468
        video_length = int(container.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        if video_length <= 0:
            video_length = None
        return video_length

    def _includes_frame(self, i):
        if self._start_frame <= i:
            if (i - self._start_frame) % self._step == 0:
                if self._end_frame is None or i < self._end_frame:
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
