import os.path as osp

import cv2
import numpy as np
import pytest

from datumaro.components.media import Video
from datumaro.util.scope import on_exit_do, scoped
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


@scoped
def make_sample_video(path, frames=4, frame_size=(10, 20), fps=25.0):
    """
    frame_size is (H, W), only even sides
    """

    writer = cv2.VideoWriter(path, frameSize=tuple(frame_size[::-1]),
        fps=float(fps), fourcc=cv2.VideoWriter_fourcc(*'MJPG'))
    on_exit_do(writer.release)

    for i in range(frames):
        # Apparently, only uint8 values are supported, but not floats
        # Colors are compressed, but grayscale colors suffer no loss
        writer.write(np.ones((*frame_size, 3), dtype=np.uint8) * i)

@pytest.fixture(scope='module')
def fxt_sample_video():
    with TestDir() as test_dir:
        video_path = osp.join(test_dir, 'video.avi')
        make_sample_video(video_path, frame_size=(4, 6))

        yield video_path

class VideoTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_video(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        assert None is video.length
        assert (4, 6) == video.frame_size

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames_sequentially(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        for idx, frame in enumerate(video):
            assert frame.size == video.frame_size
            assert frame.index == idx
            assert frame.video is video
            assert np.array_equal(frame.data,
                np.ones((*video.frame_size, 3)) * idx)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames_randomly(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        for idx in {1, 3, 2, 0, 3}:
            frame = video[idx]
            assert frame.index == idx
            assert np.array_equal(frame.data,
                np.ones((*video.frame_size, 3)) * idx)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_frames_between(self, fxt_sample_video):
        video = Video(fxt_sample_video, step=2)
        on_exit_do(video.close)

        for idx, frame in enumerate(video):
            assert 2 * idx == frame.index

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_from_start(self, fxt_sample_video):
        video = Video(fxt_sample_video, start_frame=1)
        on_exit_do(video.close)

        assert 1 == next(iter(video)).index

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_from_end(self, fxt_sample_video):
        video = Video(fxt_sample_video, end_frame=2)
        on_exit_do(video.close)

        last_frame = None
        for last_frame in video:
            pass

        assert 2 == video.length
        assert 1 == last_frame.index

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_init_frame_count_lazily(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        assert None is video.length

        for idx, frame in enumerate(video):
            assert idx == frame.index

        assert 4 == video.length

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_open_lazily(self):
        with TestDir() as test_dir:
            video = Video(osp.join(test_dir, 'path.mp4'))

            assert osp.join(test_dir, 'path.mp4') == video.path
            assert '.mp4' == video.ext
