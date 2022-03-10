import os.path as osp
from unittest import TestCase

import cv2
import numpy as np
import pytest

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image, Video
from datumaro.components.media_manager import MediaManager
from datumaro.components.project import Project
from datumaro.util.scope import Scope, on_exit_do, scope_add, scoped
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement


@scoped
def make_sample_video(path, frames=4, frame_size=(10, 20), fps=25.0):
    """
    frame_size is (H, W), only even sides
    """

    writer = cv2.VideoWriter(
        path,
        frameSize=tuple(frame_size[::-1]),
        fps=float(fps),
        fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
    )
    on_exit_do(writer.release)

    for i in range(frames):
        # Apparently, only uint8 values are supported, but not floats
        # Colors are compressed, but grayscale colors suffer no loss
        writer.write(np.ones((*frame_size, 3), dtype=np.uint8) * i)


@pytest.fixture(scope="module")
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
                    "frame_%03d" % i, subset="train", media=Image(data=np.ones((4, 6, 3)) * i)
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
                DatasetItem("frame_%06d" % i, media=Image(data=np.ones((4, 6, 3)) * i))
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
