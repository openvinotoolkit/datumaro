# Copyright (C) 2021-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import os
import os.path as osp
import shutil
import weakref
from enum import IntEnum
from typing import Any, Callable, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import cv2
import numpy as np

from datumaro.components.crypter import NULL_CRYPTER, Crypter
from datumaro.components.errors import MediaShapeError
from datumaro.util.definitions import BboxIntCoords
from datumaro.util.image import (
    _image_loading_errors,
    copyto_image,
    decode_image,
    lazy_image,
    save_image,
)

AnyData = TypeVar("AnyData", bytes, np.ndarray)


class MediaType(IntEnum):
    NONE = 0
    MEDIA_ELEMENT = 1
    IMAGE = 2
    BYTE_IMAGE = 3
    VIDEO_FRAME = 4
    VIDEO = 5
    POINT_CLOUD = 6
    MULTIFRAME_IMAGE = 7
    ROI_IMAGE = 8
    MOSAIC_IMAGE = 9


class MediaElement:
    _type = MediaType.MEDIA_ELEMENT

    def __init__(self, crypter: Crypter = NULL_CRYPTER) -> None:
        self._crypter = crypter

    @property
    def is_encrypted(self) -> bool:
        return not self._crypter.is_null_crypter

    def set_crypter(self, crypter: Crypter):
        self._crypter = crypter

    @property
    def type(self) -> MediaType:
        return self._type

    @property
    def data(self) -> Optional[Any]:
        raise NotImplementedError

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        other_type = getattr(other, "type", None)
        if self.type != other_type:
            return False
        return self.data == other.data


class MediaElementFromFileMixin:
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert path, "Path can't be empty"
        self._path = path

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension (with the leading dot)"""
        return osp.splitext(osp.basename(self.path))[1]

    @property
    def data(self) -> Optional[bytes]:
        data = None
        if os.path.exists(self.path):
            with open(self.path, "wb") as f:
                data = f.read()
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self._path})"


class MediaElementFromDataMixin(Generic[AnyData]):
    def __init__(self, data: Union[Callable[[], AnyData], AnyData], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data

    @property
    def data(self) -> Optional[AnyData]:
        if callable(self._data):
            return self._data()
        return self._data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self._data})"


class Image(MediaElement):
    _type = MediaType.IMAGE

    def __init__(
        self,
        data: Union[np.ndarray, Callable[[str], np.ndarray], None] = None,
        *,
        path: Optional[str] = None,
        ext: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        crypter: Crypter = NULL_CRYPTER,
    ) -> None:
        """
        Creates an image.

        Any combination of the `data`, `path` and `size` is possible,
        but at least one of these arguments must be provided.
        The `ext` parameter cannot be used as a single argument for
        construction.

        Args:
            data: Image pixels or a function to retrieve them. The expected
                image shape is (H, W [, C]). If a function is provided,
                it must accept image path as the first argument.
            path: Image path
            ext: Image extension. Cannot be used together with `path`. It can
                be used for saving with a custom extension - in that case,
                the image need to have the `data` and `ext` fields defined.
            size: A pair (H, W), which represents image size.
        """

        if size is not None:
            assert (
                len(size) == 2 and 0 < size[0] and 0 < size[1]
            ), f"Invalid image size info '{size}'"
            size = tuple(map(int, size))
        self._size = size  # (H, W)

        if path is None:
            path = ""
        elif path:
            path = path.replace("\\", "/")
        self._path = path

        if ext:
            assert not path, "Can't specify both 'path' and 'ext' for image"

            if not ext.startswith("."):
                ext = "." + ext
            ext = ext.lower()
        else:
            ext = None
        self._ext = ext

        self._crypter = crypter

        if not isinstance(data, np.ndarray):
            assert path or callable(data) or size, "Image can not be empty"
            assert data is None or callable(data), f"Image data has unexpected type '{type(data)}'"
            if data or path and osp.isfile(path):
                data = lazy_image(path, loader=data, crypter=self._crypter)
        self._data = data

    @property
    def data(self) -> np.ndarray:
        """Image data in BGR HWC [0; 255] (float) format"""

        if callable(self._data):
            data = self._data()
        else:
            data = self._data

        if self._size is None and data is not None:
            if not 2 <= data.ndim <= 3:
                raise MediaShapeError("An image should have 2 (gray) or 3 (rgb) dims.")
            self._size = tuple(map(int, data.shape[:2]))
        return data

    @property
    def has_data(self) -> bool:
        return self._data is not None

    @property
    def has_size(self) -> bool:
        """Indicates that size info is cached and won't require image loading"""
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

    @property
    def ext(self) -> str:
        """Media file extension"""
        if self._ext is not None:
            return self._ext
        else:
            return osp.splitext(osp.basename(self.path))[1]

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return (
            (np.array_equal(self.size, other.size))
            and (self.has_data == other.has_data)
            and (self.has_data and np.array_equal(self.data, other.data) or not self.has_data)
        )

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        cur_path = osp.abspath(self.path)
        path = osp.abspath(path)

        cur_ext = self.ext.lower()
        new_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if cur_ext == new_ext and osp.isfile(cur_path):
            copyto_image(
                src_path=cur_path, dst_path=path, src_crypter=self._crypter, dst_crypter=crypter
            )
        else:
            save_image(path, self.data, crypter=crypter)

    def set_crypter(self, crypter: Crypter):
        super().set_crypter(crypter)
        if isinstance(self._data, lazy_image):
            self._data._crypter = crypter

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path


class ByteImage(Image):
    _type = MediaType.BYTE_IMAGE

    _FORMAT_MAGICS = (
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff", ".jpg"),
        (b"BM", ".bmp"),
    )

    def __init__(
        self,
        data: Union[bytes, Callable[[str], bytes], None] = None,
        *,
        path: Optional[str] = None,
        ext: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        crypter: Crypter = NULL_CRYPTER,
    ):
        if not isinstance(data, bytes):
            assert path or callable(data), "Image can not be empty"
            assert data is None or callable(data)
            if path and osp.isfile(path) or data:
                data = lazy_image(path, loader=data)

        self._bytes_data = data

        if ext is None and path is None and isinstance(data, bytes):
            ext = self._guess_ext(data)

        super().__init__(
            path=path, ext=ext, size=size, data=lambda _: decode_image(self.get_bytes())
        )
        if data is None:
            # We don't expect decoder to produce images from nothing,
            # otherwise using this class makes no sense. We undefine
            # data to avoid using default image loader for loading binaries
            # from the path, when no data is provided.
            self._data = None

        self._crypter = crypter

    @classmethod
    def _guess_ext(cls, data: bytes) -> Optional[str]:
        return next(
            (ext for magic, ext in cls._FORMAT_MAGICS if data.startswith(magic)),
            None,
        )

    def get_bytes(self):
        if callable(self._bytes_data):
            return self._bytes_data()
        return self._bytes_data

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        if not crypter.is_null_crypter:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement save() with non NullCrypter."
            )

        cur_path = osp.abspath(self.path)
        path = osp.abspath(path)

        cur_ext = self.ext.lower()
        new_ext = osp.splitext(osp.basename(path))[1].lower()

        os.makedirs(osp.dirname(path), exist_ok=True)
        if cur_ext == new_ext and osp.isfile(cur_path):
            if cur_path != path:
                shutil.copyfile(cur_path, path)
        elif cur_ext == new_ext:
            with open(path, "wb") as f:
                f.write(self.get_bytes())
        else:
            save_image(path, self.data)


class VideoFrame(Image):
    _type = MediaType.VIDEO_FRAME

    def __init__(self, video: Video, index: int):
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
    def video(self) -> Video:
        return self._video


class _VideoFrameIterator(Iterator[VideoFrame]):
    """
    Provides sequential access to the video frames.
    """

    _video: Video
    _iterator: Iterator[VideoFrame]
    _pos: int
    _current_frame_data: Optional[np.ndarray]

    def __init__(self, video: Video):
        self._video = video
        self._reset()

    def _reset(self):
        self._video._reset_reader()
        self._iterator = self._decode(self._video._get_reader())
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
                yield self._make_frame(index=self._pos)

            success, frame = cap.read()

        if self._video._frame_count is None:
            self._video._frame_count = self._pos + 1

    def _make_frame(self, index) -> VideoFrame:
        return VideoFrame(self._video, index=index)

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
            v = self._make_frame(index=self._pos)

        return v


class Video(MediaElement, Iterable[VideoFrame]):
    _type = MediaType.VIDEO

    """
    Provides random access to the video frames.
    """

    def __init__(
        self, path: str, *, step: int = 1, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> None:
        super().__init__()
        self._path = path

        if end_frame:
            assert start_frame < end_frame
        assert 0 < step
        self._step = step
        self._start_frame = start_frame
        self._end_frame = end_frame or None

        self._reader = None
        self._iterator: Optional[_VideoFrameIterator] = None
        self._frame_size: Optional[Tuple[int, int]] = None

        # We don't provide frame count unless we have a reliable source of
        # this information.
        # - Not all videos provide length / duration metainfo
        # - We can get an estimation based on metadata, but it
        #   can be invalid or inaccurate due to variable frame rate
        #   or fractional values rounded up. Relying on the value will give
        #   errors during requesting frames.
        # https://stackoverflow.com/a/47796468
        self._frame_count = None
        self._length = None

        from .media_manager import MediaManager

        MediaManager.get_instance().push(weakref.ref(self), self)

    def close(self):
        self._iterator = None

        if self._reader is not None:
            self._reader.release()
            self._reader = None

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
    def length(self) -> Optional[int]:
        """
        Returns frame count, if video provides such information.

        Note that not all videos provide length / duration metainfo, so the
        result may be undefined.

        Also note, that information may be inaccurate because of variable
        FPS in video or incorrect metainfo. The count is only guaranteed to
        be valid after video is completely read once.

        The count is affected by the frame filtering options of the object,
        i.e. start frame, end frame and frame step.
        """

        if self._length is None:
            end_frame = self._get_end_frame()

            length = None
            if end_frame is not None:
                length = (end_frame - self._start_frame) // self._step
                assert 0 < length

            self._length = length

        return self._length

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Returns (H, W)"""

        if self._frame_size is None:
            self._frame_size = self._get_frame_size()
        return self._frame_size

    def _get_frame_size(self) -> Tuple[int, int]:
        cap = self._get_reader()
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if h and w:
            frame_size = (int(h), int(w))
        else:
            image = next(self._get_iterator()).data
            frame_size = image.shape[0:2]

        return frame_size

    def _get_end_frame(self):
        if self._end_frame is not None and self._frame_count is not None:
            end_frame = min(self._end_frame, self._frame_count)
        else:
            end_frame = self._end_frame or self._frame_count

        return end_frame

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

    def _get_reader(self):
        if self._reader is None:
            self._reset_reader()
        return self._reader

    def _reset_reader(self):
        if self._reader is not None:
            self._reader.release()
        self._reader = cv2.VideoCapture(self._path)
        assert self._reader.isOpened()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, __class__):
            return False

        return (
            self.path == other.path
            and self._start_frame == other._start_frame
            and self._step == other._step
            and self._end_frame == other._end_frame
        )

    def __hash__(self):
        # Required for caching
        return hash((self._path, self._step, self._start_frame, self._end_frame))

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension (with the leading dot)"""
        return osp.splitext(osp.basename(self.path))[1]


class PointCloud(MediaElement):
    _type = MediaType.POINT_CLOUD

    def __init__(
        self,
        extra_images: Optional[List[Image], Callable[[], List[Image]]] = None,
        crypter: Crypter = NULL_CRYPTER,
    ):
        assert self.__class__ != PointCloud, (
            f"Directly initalizing {self.__class__.__name__} is not supported. "
            f"Please use fractory function '{self.__class__.__name__}.from_file()' "
            f"or '{self.__class__.__name__}.from_data()'."
        )
        super().__init__(crypter)
        self._extra_images = extra_images or []

    @classmethod
    def from_file(cls, path: str, *args, **kwargs):
        return PointCloudFromFile(path, *args, **kwargs)

    @classmethod
    def from_data(cls, data: Union[bytes, Callable[[], bytes]], *args, **kwargs):
        return PointCloudFromData(data, *args, **kwargs)

    @property
    def extra_images(self) -> List[Image]:
        if callable(self._extra_images):
            extra_images = self._extra_images()
            assert isinstance(extra_images, list) and all(
                [isinstance(image, Image) for image in extra_images]
            )
            return extra_images
        return self._extra_images

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self.extra_images == other.extra_images


class PointCloudFromFile(MediaElementFromFileMixin, PointCloud):
    @property
    def data(self) -> Optional[bytes]:
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                bytes_data = f.read()
            return bytes_data
        return None

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        if not crypter.is_null_crypter:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement save() with non NullCrypter."
            )

        cur_path = osp.abspath(self.path) if self.path else None
        path = osp.abspath(path)

        if cur_path is not None and osp.isfile(cur_path):
            if cur_path != path:
                os.makedirs(osp.dirname(path), exist_ok=True)
                shutil.copyfile(cur_path, path)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cur_path)


class PointCloudFromData(MediaElementFromDataMixin[bytes], PointCloud):
    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        if not crypter.is_null_crypter:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement save() with non NullCrypter."
            )

        path = osp.abspath(path)
        os.makedirs(osp.dirname(path), exist_ok=True)
        data = self.data
        with open(path, "wb") as f:
            f.write(data)


class MultiframeImage(MediaElement):
    _type = MediaType.MULTIFRAME_IMAGE

    def __init__(
        self,
        images: Optional[Iterable[Union[str, Image, np.ndarray, Callable[[str], np.ndarray]]]],
        *,
        path: Optional[str] = None,
    ):
        self._path = path

        if images is None:
            images = []

        self._images = [None] * len(images)
        for i, image in enumerate(images):
            assert isinstance(image, (str, Image, np.ndarray)) or callable(image)

            if isinstance(image, str):
                image = Image(path=image)
            elif isinstance(image, np.ndarray) or callable(image):
                image = Image(data=image)

            self._images[i] = image

        assert self._path or self._images

    @property
    def data(self) -> List[Image]:
        return self._images

    @property
    def path(self) -> str:
        """Path to the media file"""
        return self._path

    @property
    def ext(self) -> str:
        """Media file extension (with the leading dot)"""
        return osp.splitext(osp.basename(self.path))[1]


class RoIImage(Image):
    _type = MediaType.ROI_IMAGE

    def __init__(
        self,
        data: Union[np.ndarray, Callable[[str], np.ndarray], None] = None,
        *,
        path: Optional[str] = None,
        ext: Optional[str] = None,
        roi: BboxIntCoords,
    ) -> None:
        assert len(roi) == 4 and all(isinstance(v, int) for v in roi)
        self._roi = roi
        _, _, w, h = self._roi
        super().__init__(data=data, path=path, ext=ext, size=(h, w))

    @classmethod
    def create_from_image(cls, img: Image, roi: BboxIntCoords) -> RoIImage:
        if not isinstance(img, Image):
            raise TypeError(f"type(img)={type(img)} should be Image.")

        data = lambda _: img._data() if isinstance(img._data, lazy_image) else img._data
        return cls(data=data, path=img._path, ext=img._ext, roi=roi)

    @property
    def roi(self) -> BboxIntCoords:
        return self._roi

    @property
    def data(self) -> np.ndarray:
        """Image data in BGR HWC [0; 255] (float) format"""
        x, y, w, h = self._roi
        img = super().data
        return img[y : y + h, x : x + w]

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        if not crypter.is_null_crypter:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement save() with non NullCrypter."
            )
        path = osp.abspath(path)
        os.makedirs(osp.dirname(path), exist_ok=True)
        save_image(path, self.data)


ImageWithRoI = Tuple[Image, BboxIntCoords]


class MosaicImage(Image):
    _type = MediaType.MOSAIC_IMAGE

    def __init__(self, imgs: List[ImageWithRoI], size: Tuple[int, int]) -> None:
        def _get_mosaic_img(_) -> np.ndarray:
            h, w = self.size
            mosaic_img = np.zeros(shape=(h, w, 3), dtype=np.uint8)
            for img, roi in imgs:
                assert isinstance(img, Image), "MosaicImage can only take a list of Images."
                x, y, w, h = roi
                mosaic_img[y : y + h, x : x + w] = img.data
            return mosaic_img

        super().__init__(data=_get_mosaic_img, path=None, ext=None, size=size)

    def save(self, path, crypter: Crypter = NULL_CRYPTER):
        if not crypter.is_null_crypter:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement save() with non NullCrypter."
            )
        path = osp.abspath(path)
        os.makedirs(osp.dirname(path), exist_ok=True)
        save_image(path, self.data)
