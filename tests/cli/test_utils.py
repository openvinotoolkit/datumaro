from unittest.case import TestCase
import os

import numpy as np

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DatasetItem
from datumaro.components.media_manager import MediaManager
from datumaro.util.scope import on_exit_do, scope_add, scoped
from datumaro.util.test_utils import TestDir, compare_datasets
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement
from ..test_video import fxt_sample_video  # pylint: disable=unused-import


class VideoSplittingTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_split_video(self, fxt_sample_video):
        on_exit_do(MediaManager.get_instance().clear)

        test_dir = scope_add(TestDir())

        run(TestCase(), 'util', 'split_video',
            '-i', fxt_sample_video, '-o', test_dir,
            '--image-ext', '.jpg', '--end-frame', '4')

        assert set(os.listdir(test_dir)) == {'%06d.jpg' % n for n in range(4)}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_split_and_load(self, fxt_sample_video):
        test_dir = scope_add(TestDir())
        on_exit_do(MediaManager.get_instance().clear)

        expected = Dataset.from_iterable([
            DatasetItem('frame_%06d' % i, image=np.ones((4, 6, 3)) * i)
            for i in range(4)
        ])

        dataset = Dataset.import_from(fxt_sample_video, 'video_frames',
            start_frame=0, end_frame=4, name_pattern='frame_%06d')
        dataset.export(format='image_dir', save_dir=test_dir,
            image_ext='.jpg')

        actual = Dataset.import_from(test_dir, 'image_dir')
        compare_datasets(TestCase(), expected, actual)
