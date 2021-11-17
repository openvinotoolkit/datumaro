from unittest import TestCase
import os.path as osp

import cv2
import numpy as np

from datumaro.components.media import Video
from datumaro.util.scope import on_exit_do, scope_add, scoped
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

class VideoTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames(self):
        test_dir = scope_add(TestDir())
        video_path = osp.join(test_dir, 'video.avi')
        make_sample_video(video_path, frame_size=(4, 6))

        video = Video(video_path)
        on_exit_do(video.close)

        self.assertEqual(4, video.frame_count)
        self.assertEqual((4, 6), video.frame_size)

        for idx, frame in enumerate(video):
            self.assertEqual(frame.size, video.frame_size)
            self.assertEqual(frame.index, idx)
            self.assertEqual(frame.video, video)
            self.assertEqual(frame, np.ones((*video.frame_size, 3)) * 255)
