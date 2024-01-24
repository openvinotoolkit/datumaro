# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import filecmp
import os.path as osp
from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import Bbox, Label
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.media import Image, Video, VideoFrame
from datumaro.components.media_manager import MediaManager
from datumaro.components.project import Project
from datumaro.util.scope import Scope, on_exit_do, scope_add, scoped

from ..requirements import Requirements, mark_requirement
from ..utils.video import make_sample_video

from tests.utils.test_utils import TestDir, compare_datasets


@pytest.fixture()
def fxt_sample_video():
    with TestDir() as test_dir:
        video_path = osp.join(test_dir, "video.avi")
        make_sample_video(video_path, frame_size=(4, 6), frames=4)

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
            assert np.array_equal(frame.data, np.ones((*video.frame_size, 3)) * idx)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames_randomly(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        for idx in {1, 3, 2, 0, 3}:
            frame = video[idx]
            assert frame.index == idx
            assert np.array_equal(frame.data, np.ones((*video.frame_size, 3)) * idx)

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
            video = Video(osp.join(test_dir, "path.mp4"))

            assert osp.join(test_dir, "path.mp4") == video.path
            assert ".mp4" == video.ext


class VideoExtractorTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_read_frames(self, fxt_sample_video):
        on_exit_do(MediaManager.get_instance().clear)

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    "frame_%03d" % i,
                    subset="train",
                    media=Image.from_numpy(data=np.ones((4, 6, 3)) * i),
                )
                for i in range(4)
            ]
        )

        actual = Dataset.import_from(
            fxt_sample_video, "video_frames", subset="train", name_pattern="frame_%03d"
        )

        compare_datasets(TestCase(), expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_split_and_load(self, fxt_sample_video):
        test_dir = scope_add(TestDir())
        on_exit_do(MediaManager.get_instance().clear)

        expected = Dataset.from_iterable(
            [
                DatasetItem("frame_%06d" % i, media=Image.from_numpy(data=np.ones((4, 6, 3)) * i))
                for i in range(4)
            ]
        )

        dataset = Dataset.import_from(
            fxt_sample_video, "video_frames", start_frame=0, end_frame=4, name_pattern="frame_%06d"
        )
        dataset.export(format="image_dir", save_dir=test_dir, image_ext=".jpg")

        actual = Dataset.import_from(test_dir, "image_dir")
        compare_datasets(TestCase(), expected, actual)


class ProjectTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_release_resources_on_exit(self, fxt_sample_video):
        with Scope() as scope:
            test_dir = scope.add(TestDir())

            project = scope.add(Project.init(test_dir))

            project.import_source(
                "src",
                osp.dirname(fxt_sample_video),
                "video_frames",
                rpath=osp.basename(fxt_sample_video),
            )

            assert len(project.working_tree.make_dataset()) == 4
        assert not osp.exists(test_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_release_resources_on_remove(self, fxt_sample_video):
        test_dir = scope_add(TestDir())

        project = scope_add(Project.init(test_dir))

        project.import_source(
            "src",
            osp.dirname(fxt_sample_video),
            "video_frames",
            rpath=osp.basename(fxt_sample_video),
        )
        project.commit("commit 1")

        assert len(project.working_tree.make_dataset()) == 4
        assert osp.isdir(osp.join(test_dir, "src"))

        project.remove_source("src", keep_data=False)

        assert not osp.exists(osp.join(test_dir, "src"))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_release_resources_on_checkout(self, fxt_sample_video):
        test_dir = scope_add(TestDir())

        project = scope_add(Project.init(test_dir))

        src_url = osp.join(test_dir, "src")
        src = Dataset.from_iterable(
            [
                DatasetItem(1),
            ],
            categories=["a"],
        )
        src.save(src_url)
        project.add_source(src_url, "datumaro")
        project.commit("commit 1")

        project.remove_source("src", keep_data=False)

        project.import_source(
            "src",
            osp.dirname(fxt_sample_video),
            "video_frames",
            rpath=osp.basename(fxt_sample_video),
        )
        project.commit("commit 2")

        assert len(project.working_tree.make_dataset()) == 4
        assert osp.isdir(osp.join(test_dir, "src"))

        project.checkout("HEAD~1")

        assert len(project.working_tree.make_dataset()) == 1


@pytest.mark.new
class VideoAnnotationTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("dataset_format", ["datumaro", "datumaro_binary"])
    @scoped
    def test_can_video_annotation_export(self, dataset_format, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    i,
                    media=VideoFrame(video, i),
                    annotations=[Label(i % 2), Bbox(1, 2, 3, 4, object_id=1)],
                    subset="train" if i < 3 else "test",
                )
                for i in range(4)
            ],
            categories=["a", "b"],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "test_video")
            expected.export(dataset_path, dataset_format, save_media=True)
            actual = Dataset.import_from(dataset_path)
            compare_datasets(TestCase(), expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_save_video(self, fxt_sample_video):
        video = Video(fxt_sample_video)
        on_exit_do(video.close)

        with TestDir() as test_dir:
            path_video_save_by_path = osp.join(test_dir, "test_video_save_path.avi")
            video.save(path_video_save_by_path)
            assert filecmp.cmp(video.path, path_video_save_by_path)

            path_video_save_by_io = osp.join(test_dir, "test_video_save_io.avi")
            with open(path_video_save_by_io, "wb") as f_io:
                video.save(f_io)
            assert filecmp.cmp(video.path, path_video_save_by_io)
