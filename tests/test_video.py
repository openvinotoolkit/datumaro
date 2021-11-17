from unittest import mock
import os.path as osp

import cv2
import numpy as np
import pytest

from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.media import Video, VideoFrame
from datumaro.util.scope import on_exit_do, scoped
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


def make_sample_video(path, frames=4, frame_size=(10, 20), fps=25.0):
    """
    frame_size is (H, W), only even sides
    """

    writer = cv2.VideoWriter(path, frameSize=tuple(frame_size[::-1]),
        fps=float(fps), fourcc=cv2.VideoWriter_fourcc(*'MJPG'))

    for _ in range(frames):
        # Apparently, only uint8 values are supported, but not floats
        writer.write(np.ones((*frame_size, 3), dtype=np.uint8) * 255)

    writer.release()

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

        assert 4 == video.frame_count
        assert (4, 6) == video.frame_size

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames_sequentially(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        for idx, frame in enumerate(video):
            assert frame.size == video.frame_size
            assert frame.index == idx
            assert id(frame.video) == id(video)
            assert frame == np.ones((*video.frame_size, 3)) * 255

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames_randomly(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        for idx in {1, 3, 2, 0, 3}:
            frame = video[idx]
            assert frame.index == idx
            assert frame == np.ones((*video.frame_size, 3)) * 255

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_frames_between(self, fxt_sample_video):
        video = Video(fxt_sample_video, step=2)
        on_exit_do(video.close)

        assert 2 == video.frame_count

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_from_start(self, fxt_sample_video):
        video = Video(fxt_sample_video, start_frame=1)
        on_exit_do(video.close)

        assert 3 == video.frame_count
        assert 1 == next(iter(video)).index

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_skip_from_end(self, fxt_sample_video):
        video = Video(fxt_sample_video, end_frame=2)
        on_exit_do(video.close)

        last_frame = None
        for last_frame in video:
            pass

        assert 2 == video.frame_count
        assert 1 == last_frame.index

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_init_frame_count_lazily(self, fxt_sample_video):
        with mock.patch.object(Video, '_get_frame_count',
                mock.MagicMock(return_value=None)):
            video = Video(fxt_sample_video)
        on_exit_do(video.close)

        assert None == video.frame_count

        for idx, frame in enumerate(video):
            assert idx == frame.index

        assert 4 == video.frame_count

class VideoFramesExtractorTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load(self, fxt_sample_video):
        dataset = Dataset.import_from(fxt_sample_video, 'video_frames')

        assert 4 == len(dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_with_custom_frame_names(self, fxt_sample_video):
        dataset = Dataset.import_from(fxt_sample_video, 'video_frames',
            name_pattern='custom_name-%03d')

        for idx, item in enumerate(dataset):
            assert item.id == 'custom_name-%03d' % idx

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_with_custom_subset(self, fxt_sample_video):
        dataset = Dataset.import_from(fxt_sample_video, 'video_frames',
            subset='custom')

        for item in dataset:
            assert item.subset == 'custom'

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_provide_frame_data(self, fxt_sample_video):
        dataset = Dataset.import_from(fxt_sample_video, 'video_frames')

        for item in dataset:
            video = item.media_as(VideoFrame).video
            assert item.media == np.ones((*video.frame_size, 3)) * 255

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self, fxt_sample_video):
        assert 'video_frames' in Environment().detect_dataset(
            osp.dirname(fxt_sample_video))
