# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from unittest import TestCase

import pytest

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.errors import DatasetImportError
from datumaro.components.media import MediaElement, Video, VideoFrame

from ...requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import TestDir, compare_datasets
from tests.utils.test_utils import run_datum as run
from tests.utils.video import make_sample_video


class VideoTest(TestCase):
    video_dir = get_test_asset_path("video_dataset")

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_display_video_import_warning_in_import(self):
        with TestDir() as test_dir:
            video_dir = osp.join(test_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=4)

            proj_dir = osp.join(test_dir, "proj")
            run(self, "project", "create", "-o", proj_dir)

            with self.assertLogs() as capture:
                run(
                    self,
                    "project",
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
            run(self, "project", "create", "-o", proj_dir)

            video_dir = osp.join(proj_dir, "src")
            os.makedirs(video_dir)
            make_sample_video(osp.join(video_dir, "video.avi"), frames=4)

            with self.assertLogs() as capture:
                run(
                    self,
                    "project",
                    "add",
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
    def test_can_extract_frames_from_video(self):
        expected = Dataset.from_iterable([DatasetItem("000000"), DatasetItem("000002")])

        with TestDir() as test_dir:
            proj_dir = osp.join(test_dir, "proj")
            run(self, "project", "create", "-o", proj_dir)

            run(
                self,
                "project",
                "import",
                "-f",
                "video_frames",
                "-p",
                proj_dir,
                "-r",
                "video.avi",
                self.video_dir,
                "--start-frame",
                "0",
                "--end-frame",
                "3",
                "--step",
                "2",
            )

            result_dir = osp.join(proj_dir, "result")
            run(self, "project", "export", "-f", "image_dir", "-p", proj_dir, "-o", result_dir)
            parsed_dataset = Dataset.import_from(result_dir, "image_dir")

            compare_datasets(self, expected, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_extract_keyframes_from_video(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem("000000"),
                DatasetItem("000001"),
                DatasetItem("000004"),
                DatasetItem("000005"),
            ]
        )

        with TestDir() as test_dir:
            proj_dir = osp.join(test_dir, "proj")
            run(self, "project", "create", "-o", proj_dir)

            run(
                self,
                "project",
                "import",
                "-f",
                "video_keyframes",
                "-p",
                proj_dir,
                "-r",
                "video.avi",
                self.video_dir,
                "--threshold",
                "0.3",
            )

            result_dir = osp.join(proj_dir, "result")
            run(self, "project", "export", "-f", "image_dir", "-p", proj_dir, "-o", result_dir)
            parsed_dataset = Dataset.import_from(result_dir, "image_dir")

            compare_datasets(self, expected, parsed_dataset)

    @staticmethod
    def _make_sample_video_dataset(test_dir: str, media_type=None) -> Dataset:
        video_path = osp.join(test_dir, "video.avi")
        make_sample_video(video_path, frame_size=(4, 6), frames=4)
        video = Video(video_path)

        kwargs = {
            "iterable": [
                DatasetItem(
                    "0",
                    media=VideoFrame(video, 0),
                    annotations=[
                        Bbox(1, 1, 1, 1, label=0, object_id=0),
                        Bbox(2, 2, 2, 2, label=1, object_id=1),
                    ],
                ),
                DatasetItem(
                    "1",
                    media=Video(video_path, step=1, start_frame=0, end_frame=0),
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    "2",
                    media=Video(video_path, step=1, start_frame=1, end_frame=2),
                    annotations=[
                        Label(1),
                    ],
                ),
            ],
            "categories": ["a", "b"],
        }
        if media_type:
            kwargs["media_type"] = media_type
        dataset = Dataset.from_iterable(**kwargs)

        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cannot_create_video_and_videoframe_dataset_with_wrong_media_type(self):
        with TestDir() as test_dir:
            dataset = self._make_sample_video_dataset(test_dir)
            project_dir = osp.join(test_dir, "proj")
            run(self, "project", "create", "-o", project_dir)
            with self.assertRaises(DatasetImportError):
                dataset_dir = osp.join(test_dir, "test_video")
                dataset.save(dataset_dir, save_media=True)
                run(self, "project", "import", "-p", project_dir, "-f", "datumaro", dataset_dir)

    @pytest.mark.new
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_export_video_dataset(self):
        with TestDir() as test_dir:
            expected = self._make_sample_video_dataset(test_dir, media_type=MediaElement)

            project_dir = osp.join(test_dir, "proj")
            run(self, "project", "create", "-o", project_dir)

            dataset_dir = osp.join(test_dir, "test_video")
            expected.save(dataset_dir, save_media=True)

            run(self, "project", "import", "-p", project_dir, "-f", "datumaro", dataset_dir)

            result_dir = osp.join(test_dir, "test_video_result")
            run(
                self,
                "project",
                "export",
                "-p",
                project_dir,
                "-f",
                "datumaro",
                "-o",
                result_dir,
                "--",
                "--save-media",
            )
            actual = Dataset.import_from(result_dir, "datumaro")

            compare_datasets(self, actual, expected)
