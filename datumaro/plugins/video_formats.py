# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

import av
import cv2

from datumaro.components.extractor import (
    DEFAULT_SUBSET_NAME, DatasetItem, Extractor, Importer,
)
from datumaro.util.image import Image


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


class VideoFramesImporter(Importer):
    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('--subset')
        parser.add_argument('-p', '--name-pattern', default='%06d')
        parser.add_argument('-s', '--step', type=int, default=1)
        parser.add_argument('-b', '--start-frame', type=int, default=0)
        parser.add_argument('-e', '--end-frame', type=int, default=None)
        return parser

    @classmethod
    def find_sources(cls, path):
        if not osp.isfile(path):
            return []
        return [{ 'url': path, 'format': 'video_frames' }]

class VideoFramesExtractor(Extractor):
    def __init__(self, url, subset=None, name_pattern='%06d',
            step=1, start_frame=0, end_frame=None):
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(subsets=[self._subset])

        assert osp.isfile(url), url

        self._name_pattern = name_pattern
        self._reader = VideoReader(url, step=step,
            start_frame=start_frame, end_frame=end_frame)

        duration = self._reader.get_duration()
        if duration is not None:
            self._length = \
                (min(duration, end_frame or duration) - start_frame) // step

    def __iter__(self):
        frame_iter = RandomAccessIterator(self._reader)

        seq_iter = frame_iter
        if self._length is not None:
            seq_iter = range(self._length)

        frame_size = self._reader.get_image_size()

        for frame_idx, _ in enumerate(seq_iter):
            yield DatasetItem(id=self._name_pattern % frame_idx,
                subset=self._subset,
                image=Image(loader=self._lazy_get_frame(frame_iter, frame_idx),
                    size=frame_size))

    @staticmethod
    def _lazy_get_frame(frame_iter, idx):
        return lambda _: frame_iter[idx].to_ndarray(format='bgr24')


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
    def __init__(self, path, step=1, start_frame=0, end_frame=None):
        self._path = path
        self._step = int(step) if step else None
        self._start_frame = int(start_frame)
        self._end_frame = int(end_frame) if end_frame else None

    def _includes_frame(self, i):
        if self._start_frame <= i:
            if (i - self._start_frame) % self._step == 0:
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

    def _get_av_container(self):
        container = av.open(self._path)

        # Allow independent multithreaded frame decoding
        container.streams.video[0].thread_type = 'AUTO'
        return container

    def get_duration(self):
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
                h, m, s = duration_str.split(':')
                duration_sec = 60 * 60 * float(h) + 60 * float(m) + float(s)
                duration = duration_sec * tb_denominator

        return duration

    def get_image_size(self):
        image = next(iter(self))
        return (image.width, image.height)
