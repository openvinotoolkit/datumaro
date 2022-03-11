import os
import os.path as osp
from unittest import TestCase

from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement

from tests.test_video import make_sample_video


class VideoTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_display_video_import_warning_in_import(self):
        with TestDir() as test_dir:
            video_dir = osp.join(test_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=4)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)

            with self.assertLogs() as capture:
                run(
                    self,
                    "import",
                    "-f",
                    "video_frames",
                    "-p",
                    proj_dir,
                    "-r",
                    "video.avi",
                    video_dir,
                )

            self.assertTrue("results across multiple runs" in "\n".join(capture.output))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_display_video_import_warning_in_add(self):
        with TestDir() as test_dir:
            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)

            video_dir = osp.join(proj_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=4)

            with self.assertLogs() as capture:
                run(self, "add", "-f", "video_frames", "-p", proj_dir, "-r", "video.avi", video_dir)

            self.assertTrue("results across multiple runs" in "\n".join(capture.output))
