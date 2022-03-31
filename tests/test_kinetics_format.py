import os.path as osp
from unittest import TestCase

from datumaro.components.media import Video
from datumaro.util.test_utils import compare_datasets

DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "kinetics_dataset")

from datumaro.components.annotation import Label
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.environment import Environment
from datumaro.plugins.kinetics_format import KineticsImporter

from .requirements import Requirements, mark_requirement


class KineticsImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([KineticsImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_video(self):
        attributes = {"time_start": 0, "time_end": 2}
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="test",
                    annotations=[Label(0, attributes={"time_start": 0, "time_end": 2})],
                    media=Video("./video_1.avi"),
                ),
                DatasetItem(
                    id="2",
                    subset="test",
                    annotations=[Label(0, attributes={"time_start": 5, "time_end": 7})],
                ),
                DatasetItem(
                    id="4",
                    subset="test",
                    annotations=[Label(1, attributes={"time_start": 10, "time_end": 15})],
                ),
                DatasetItem(
                    id="3",
                    subset="train",
                    annotations=[Label(2, attributes={"time_start": 0, "time_end": 2})],
                    media=Video("./train/3.avi"),
                ),
            ],
            categories=["label_0", "label_1", "label_2"],
            media_type=Video,
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "kinetics")

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)
