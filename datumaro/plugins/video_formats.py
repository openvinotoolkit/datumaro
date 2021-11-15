# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional
import os.path as osp

from datumaro.components.extractor import (
    DEFAULT_SUBSET_NAME, DatasetItem, Extractor, Importer,
)
from datumaro.components.media import Video


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
    def __init__(self, url: str, *,
            subset: Optional[str] = None, name_pattern: str = '%06d',
            step: int = 1, start_frame: int = 0,
            end_frame: Optional[int] = None) -> None:
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(subsets=[self._subset])

        assert osp.isfile(url), url

        self._name_pattern = name_pattern
        self._reader = Video(url, step=step,
            start_frame=start_frame, end_frame=end_frame)
        self._length = len(self._reader)

    def __iter__(self):
        for frame in self._reader:
            yield DatasetItem(id=self._name_pattern % frame.index,
                subset=self._subset, media=frame)

    def get(self, id, subset=None):
        assert subset == self._subset, '%s != %s' % (subset, self._subset)
        return super().get(id, subset or self._subset)
