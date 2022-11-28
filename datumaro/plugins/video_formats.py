# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import Optional

from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetBase, DatasetItem
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Video, VideoFrame
from datumaro.util.os_util import find_files

# Taken from https://en.wikipedia.org/wiki/Comparison_of_video_container_formats
# An extension does not define file contents, but it can be a good file filter
VIDEO_EXTENSIONS = [
    "3gp",
    "3g2",
    "asf",
    "wmv",
    "avi",
    "divx",
    "evo",
    "f4v",
    "flv",
    "mkv",
    "mk3d",
    "mp4",
    "mpg",
    "mpeg",
    "m2p",
    "ps",
    "ts",
    "m2ts",
    "mxf",
    "ogg",
    "ogv",
    "ogx",
    "mov",
    "qt",
    "rmvb",
    "vob",
    "webm",
]


class VideoFramesImporter(Importer):
    """
    Reads video frames as a dataset.
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--subset",
            help="The name of the subset for the produced dataset items " "(default: none)",
        )
        parser.add_argument(
            "-p",
            "--name-pattern",
            default="%06d",
            help="The name pattern for the produced dataset items " "(default: %(default)s).",
        )
        parser.add_argument(
            "-s", "--step", type=int, default=1, help="Frame step (default: %(default)s)"
        )
        parser.add_argument(
            "-b", "--start-frame", type=int, default=0, help="Starting frame (default: %(default)s)"
        )
        parser.add_argument(
            "-e",
            "--end-frame",
            type=int,
            default=None,
            help="Finishing frame (default: %(default)s)",
        )
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        try:
            next(find_files(context.root_path, VIDEO_EXTENSIONS, recursive=True))
            return FormatDetectionConfidence.LOW
        except StopIteration:
            context.fail(
                "No video files found in '%s'. "
                "Checked extensions: %s" % (context.root_path, ", ".join(VIDEO_EXTENSIONS))
            )

    @classmethod
    def find_sources(cls, path):
        if not osp.isfile(path):
            return []
        return [{"url": path, "format": VideoFramesBase.NAME}]


class VideoFramesBase(DatasetBase):
    def __init__(
        self,
        url: str,
        *,
        subset: Optional[str] = None,
        name_pattern: str = "%06d",
        step: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> None:
        self._subset = subset or DEFAULT_SUBSET_NAME
        super().__init__(subsets=[self._subset], media_type=VideoFrame)

        assert osp.isfile(url), url

        self._name_pattern = name_pattern
        self._reader = Video(url, step=step, start_frame=start_frame, end_frame=end_frame)
        self._length = self._reader.length  # NOTE: the value is often incorrect

    def __iter__(self):
        for frame in self._reader:
            yield DatasetItem(
                id=self._name_pattern % (frame.index,), subset=self._subset, media=frame
            )

    def get(self, id, subset=None):
        assert subset == self._subset, "%s != %s" % (subset, self._subset)
        return super().get(id, subset or self._subset)
