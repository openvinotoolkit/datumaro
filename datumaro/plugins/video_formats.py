# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import io
import os.path as osp

import av
import cv2

from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.util.os_util import find_files

# Taken from https://en.wikipedia.org/wiki/Comparison_of_video_container_formats
# An extension does not define file contents, but it can be a good file filter
VIDEO_EXTENSIONS = [
    '3gp', '3g2', 'asf', 'wmv', 'avi', 'divx',
    'evo', 'f4v', 'flv', 'mkv', 'mk3d', 'mp4', 'mpg', 'mpeg',
    'm2p', 'ps', 'ts', 'm2ts', 'mxf', 'ogg', 'ogv', 'ogx',
    'mov', 'qt', 'rmvb', 'vob', 'webm'
]


class RandomAccessIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = None
        self.pos = -1

    def __iter__(self):
        return self

    def __next__(self):
        return self[self.pos + 1]

    def __getitem__(self, idx):
        assert 0 <= idx
        if self.iterator is None or idx <= self.pos:
            self.reset()
        v = None
        while self.pos < idx:
            # NOTE: don't keep the last item in self, it can be expensive
            v = next(self.iterator)
            self.pos += 1
        return v

    def reset(self):
        self.iterator = iter(self.iterable)
        self.pos = -1

class VideoDirImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        return [{ 'url': path, 'format': 'video_dir' }]

class VideoDirExtractor(SourceExtractor):
    def __init__(self, url, subset=None, max_depth=None, exts=None):
        super().__init__(subset=subset)

        assert osp.isdir(url), url

        for path in find_files(url, exts=exts or VIDEO_EXTENSIONS,
                recursive=True, max_depth=max_depth):
            item_id = osp.relpath(osp.splitext(path)[0], url)
            self._items.append(DatasetItem(id=item_id, subset=self._subset,
                video=path))


class VideoFramesImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if not osp.isfile(path):
            return []
        return [{ 'url': path, 'format': 'video_frames' }]

class VideoFramesExtractor(SourceExtractor):
    def __init__(self, url, subset=None, name_pattern='%06d',
            freq=None, start_frame=0, end_frame=None):
        super().__init__(subset=subset)

        assert osp.isfile(url), url

        self._name_pattern = name_pattern
        self._reader = VideoReader(url, freq=freq,
            start_frame=start_frame, end_frame=end_frame)


def rotate_image(image, angle):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(matrix[0,0])
    abs_sin = abs(matrix[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    matrix[0, 2] += bound_w/2 - image_center[0]
    matrix[1, 2] += bound_h/2 - image_center[1]
    matrix = cv2.warpAffine(image, matrix, (bound_w, bound_h))
    return matrix

class VideoReader:
    def __init__(self, path, freq=None, start_frame=0, end_frame=None):
        self._path = path
        self._freq = int(freq) if freq else None
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame) if end_frame else None
        self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._container is not None:
            self._container.close()
            self._container = None

    def _includes_frame(self, i):
        if self._start_frame <= i:
            if (i - self._start_frame) % self._freq == 0:
                if self._end_frame is None or i < self._end_frame:
                    return True

        return False

    def _decode(self, container):
        frame_num = 0

        for packet in container.demux():
            if packet.stream.type != 'video':
                continue

            for image in packet.decode():
                frame_num += 1

                if self._includes_frame(frame_num - 1):
                    if packet.stream.metadata.get('rotate'):
                        old_image = image
                        image = av.VideoFrame().from_ndarray(
                            rotate_image(
                                image.to_ndarray(format='bgr24'),
                                360 - int(container.streams.video[0].metadata.get('rotate'))
                            ),
                            format ='bgr24'
                        )
                        image.pts = old_image.pts

                    yield image

    def __iter__(self):
        yield from self._decode(self._get_av_container())

    def get_progress(self, pos):
        duration = self._get_duration()
        return pos / duration if duration else None

    def _open_av_container(self):
        return av.open(self._path)

    def _get_av_container(self):
        if self._container is None:
            container = self._open_av_container()

            # Allow independent multithreaded frame decoding
            container.streams.video[0].thread_type = 'AUTO'

            self._container = container
        return self._container

    def _get_duration(self):
        container = self._get_av_container()
        stream = container.streams.video[0]

        duration = None
        if stream.duration:
            duration = stream.duration
        else:
            # may have a DURATION in format like "01:16:45.935000000"
            duration_str = stream.metadata.get("DURATION", None)
            tb_denominator = stream.time_base.denominator
            if duration_str and tb_denominator:
                _hour, _min, _sec = duration_str.split(':')
                duration_sec = 60*60*float(_hour) + 60*float(_min) + float(_sec)
                duration = duration_sec * tb_denominator

        return duration

    def get_preview(self):
        return next(iter(self))

    def get_image_size(self):
        image = next(iter(self))
        return image.width, image.height

class ByteVideoReader(VideoReader):
    def __init__(self, file, path=None, freq=None, start_frame=0, end_frame=None):
        super().__init__(path=path, freq=freq,
            start_frame=start_frame, end_frame=end_frame)
        self._file = file

    def _open_av_container(self):
        if isinstance(self._file, io.BytesIO):
            self._file.seek(0) # required for re-reading
        return av.open(self._file)
