import os
import os.path as osp
from unittest.case import TestCase

from datumaro.components.media_manager import MediaManager
from datumaro.util.scope import on_exit_do, scope_add, scoped
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement
from ..test_video import make_sample_video  # pylint: disable=unused-import


class VideoSplittingTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_split_video(self):
        on_exit_do(MediaManager.get_instance().clear)

        test_dir = scope_add(TestDir())
        video_path = osp.join(test_dir, "video.avi")
        make_sample_video(video_path, frames=10)

        output_dir = osp.join(test_dir, "result")

        run(
            TestCase(),
            "util",
            "split_video",
            "-i",
            video_path,
            "-o",
            output_dir,
            "--image-ext",
            ".jpg",
            "--start-frame",
            "2",
            "--end-frame",
            "8",
            "--step",
            "2",
        )

        assert set(os.listdir(output_dir)) == {"%06d.jpg" % n for n in range(2, 8, 2)}
