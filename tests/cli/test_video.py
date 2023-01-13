import os
import os.path as osp
from unittest import TestCase

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.util.test_utils import TestDir, compare_datasets
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_frames_from_video(self):
        expected = Dataset.from_iterable([DatasetItem('000000'), DatasetItem('000002')])

        with TestDir() as test_dir:
            video_dir = osp.join(test_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=10)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)

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
                "--start-frame",
                "0",
                "--end-frame",
                "4",
                "--step",
                "2",
            )

            result_dir = osp.join(proj_dir, "result")
            run(self, "export", "-f", "image_dir", "-p", proj_dir, "-o", result_dir)
            parsed_dataset = Dataset.import_from(result_dir, "image_dir")

            for d in parsed_dataset:
                print(d)

            compare_datasets(self, expected, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_keyframes_from_video(self):
        expected = Dataset.from_iterable([DatasetItem('000000')])

        with TestDir() as test_dir:
            video_dir = osp.join(test_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=10)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "create", "-o", proj_dir)

            run(
                self,
                "import",
                "-f",
                "video_keyframes",
                "-p",
                proj_dir,
                "-r",
                "video.avi",
                video_dir,
                "--threshold",
                "0.3",
            )

            result_dir = osp.join(proj_dir, "result")
            run(self, "export", "-f", "image_dir", "-p", proj_dir, "-o", result_dir)
            parsed_dataset = Dataset.import_from(result_dir, "image_dir")

            compare_datasets(self, expected, parsed_dataset)
