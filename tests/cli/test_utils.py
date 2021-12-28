from unittest.case import TestCase
import os

from datumaro.components.media_manager import MediaManager
from datumaro.util.scope import on_exit_do, scope_add, scoped
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ..test_video import fxt_sample_video  # pylint: disable=unused-import


class VideoSplittingTest:
    @scoped
    def test_can_split_video(self, fxt_sample_video):
        on_exit_do(MediaManager.get_instance().clear)

        test_dir = scope_add(TestDir())

        run(TestCase(), 'util', 'split_video',
            '-i', fxt_sample_video, '-o', test_dir,
            '--image-ext', '.jpg', '--end-frame', '4')

        assert set(os.listdir(test_dir)) == {'%06d.jpg' % n for n in range(4)}
