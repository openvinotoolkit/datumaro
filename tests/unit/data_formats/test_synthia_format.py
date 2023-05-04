from unittest import TestCase

import numpy as np
import pytest

from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.synthia.importer import SynthiaRandImporter, SynthiaSfImporter, SynthiaAlImporter

from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets

from .base import TestDataFormatBase

DUMMY_RAND_DATASET_DIR = get_test_asset_path("synthia_dataset", "rand")
DUMMY_SF_DATASET_DIR = get_test_asset_path("synthia_dataset", "sf")
DUMMY_AL_DATASET_DIR = get_test_asset_path("synthia_dataset", "al")

DUMMY_DATASET_DIR_CUSTOM_LABELMAP = get_test_asset_path(
    "synthia_dataset", "dataset_with_custom_labelmap"
)
DUMMY_DATASET_DIR_META_FILE = get_test_asset_path("synthia_dataset", "dataset_with_meta_file")


@pytest.fixture
def fxt_default_dataset():
    return Dataset.from_iterable(
        [
            DatasetItem(
                id="Stereo_Left/Omni_F/000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 1, 0, 0, 0]]),
                        label=1,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 1]]),
                        label=10,
                        attributes={"dynamic_object": True},
                    ),
                ],
            ),
            DatasetItem(
                id="Stereo_Left/Omni_F/000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 0, 0, 0, 0]]),
                        label=8,
                        attributes={"dynamic_object": True},
                    ),
                    Mask(
                        np.array([[0, 1, 1, 0, 0]]),
                        label=11,
                        attributes={"dynamic_object": True},
                    ),
                    Mask(
                        np.array([[0, 0, 0, 1, 1]]),
                        label=3,
                        attributes={"dynamic_object": False},
                    ),
                ],
            ),
            DatasetItem(
                id="Stereo_Left/Omni_F/000000",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 1, 0, 0, 0]]),
                        label=1,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 0]]),
                        label=2,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 0, 0, 0, 1]]),
                        label=3,
                        attributes={"dynamic_object": False},
                    ),
                ],
            ),
            DatasetItem(
                id="Stereo_Left/Omni_F/000001",
                media=Image.from_numpy(data=np.ones((1, 5, 3))),
                annotations=[
                    Mask(
                        np.array([[1, 0, 0, 0, 0]]),
                        label=1,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 1, 0, 0, 0]]),
                        label=2,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 0, 1, 1, 0]]),
                        label=15,
                        attributes={"dynamic_object": False},
                    ),
                    Mask(
                        np.array([[0, 0, 0, 0, 1]]),
                        label=3,
                        attributes={"dynamic_object": False},
                    ),
                ],
            ),
        ],
        # categories=Synthia.make_categories(),
    )

class SynthiaRandFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaRandImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_RAND_DATASET_DIR],
        ids=["rand"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)


class SynthiaSfFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaSfImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_SF_DATASET_DIR],
        ids=["sf"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)


class SynthiaAlFormatTest(TestDataFormatBase):
    IMPORTER = SynthiaAlImporter

    @pytest.mark.parametrize(
        "fxt_dataset_dir",
        [DUMMY_AL_DATASET_DIR],
        ids=["al"],
    )
    def test_can_detect(self, fxt_dataset_dir: str):
        return super().test_can_detect(fxt_dataset_dir)


# class SynthiaImporterTest(TestCase):
#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_detect(self):
#         detected_formats = Environment().detect_dataset(DUMMY_LABELS_SEGM_DATASET_DIR)
#         self.assertEqual([Synthia.SynthiaImporter.NAME], detected_formats)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_detect_with_colored_masks(self):
#         detected_formats = Environment().detect_dataset(DUMMY_COLOR_SEGM_DATASET_DIR)
#         self.assertEqual([Synthia.SynthiaImporter.NAME], detected_formats)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_detect_with_custom_labelmap(self):
#         detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR_CUSTOM_LABELMAP)
#         self.assertEqual([Synthia.SynthiaImporter.NAME], detected_formats)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_import(self):
#         expected_dataset = Dataset.from_iterable(
#             [
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000000",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(
#                             np.array([[1, 1, 0, 0, 0]]),
#                             label=1,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 1, 1, 1]]),
#                             label=10,
#                             attributes={"dynamic_object": True},
#                         ),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000001",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(
#                             np.array([[1, 0, 0, 0, 0]]),
#                             label=8,
#                             attributes={"dynamic_object": True},
#                         ),
#                         Mask(
#                             np.array([[0, 1, 1, 0, 0]]),
#                             label=11,
#                             attributes={"dynamic_object": True},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 0, 1, 1]]),
#                             label=3,
#                             attributes={"dynamic_object": False},
#                         ),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000000",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(
#                             np.array([[1, 1, 0, 0, 0]]),
#                             label=1,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 1, 1, 0]]),
#                             label=2,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 0, 0, 1]]),
#                             label=3,
#                             attributes={"dynamic_object": False},
#                         ),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000001",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(
#                             np.array([[1, 0, 0, 0, 0]]),
#                             label=1,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 1, 0, 0, 0]]),
#                             label=2,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 1, 1, 0]]),
#                             label=15,
#                             attributes={"dynamic_object": False},
#                         ),
#                         Mask(
#                             np.array([[0, 0, 0, 0, 1]]),
#                             label=3,
#                             attributes={"dynamic_object": False},
#                         ),
#                     ],
#                 ),
#             ],
#             categories=Synthia.make_categories(),
#         )

#         dataset = Dataset.import_from(DUMMY_LABELS_SEGM_DATASET_DIR, "synthia")

#         compare_datasets(self, expected_dataset, dataset, require_media=True)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_import_with_colored_masks(self):
#         expected_dataset = Dataset.from_iterable(
#             [
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000000",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 1, 0, 0, 0]]), label=1),
#                         Mask(np.array([[0, 0, 1, 1, 0]]), label=2),
#                         Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000001",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 0, 0, 0, 0]]), label=1),
#                         Mask(np.array([[0, 1, 0, 0, 0]]), label=2),
#                         Mask(np.array([[0, 0, 1, 1, 0]]), label=15),
#                         Mask(np.array([[0, 0, 0, 0, 1]]), label=3),
#                     ],
#                 ),
#             ],
#             categories=Synthia.make_categories(),
#         )

#         dataset = Dataset.import_from(DUMMY_COLOR_SEGM_DATASET_DIR, "synthia")

#         compare_datasets(self, expected_dataset, dataset, require_media=True)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_import_with_custom_labelmap(self):
#         expected_dataset = Dataset.from_iterable(
#             [
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000000",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 1, 1, 0, 0]]), label=1),
#                         Mask(np.array([[0, 0, 0, 1, 1]]), label=4),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000001",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 1, 0, 0, 0]]), label=2),
#                         Mask(np.array([[0, 0, 1, 1, 0]]), label=3),
#                         Mask(np.array([[0, 0, 0, 0, 1]]), label=4),
#                     ],
#                 ),
#             ],
#             categories={
#                 AnnotationType.label: LabelCategories.from_iterable(
#                     ["background", "sky", "building", "person", "road"]
#                 ),
#                 AnnotationType.mask: MaskCategories(
#                     {
#                         0: (0, 0, 0),
#                         1: (0, 0, 64),
#                         2: (0, 128, 128),
#                         3: (128, 0, 64),
#                         4: (0, 192, 128),
#                     }
#                 ),
#             },
#         )

#         dataset = Dataset.import_from(DUMMY_DATASET_DIR_CUSTOM_LABELMAP, "synthia")

#         compare_datasets(self, expected_dataset, dataset, require_media=True)

#     @mark_requirement(Requirements.DATUM_497)
#     def test_can_import_with_meta_file(self):
#         expected_dataset = Dataset.from_iterable(
#             [
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000000",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 1, 1, 0, 0]]), label=1),
#                         Mask(np.array([[0, 0, 0, 1, 1]]), label=4),
#                     ],
#                 ),
#                 DatasetItem(
#                     id="Stereo_Left/Omni_F/000001",
#                     media=Image.from_numpy(data=np.ones((1, 5, 3))),
#                     annotations=[
#                         Mask(np.array([[1, 1, 0, 0, 0]]), label=2),
#                         Mask(np.array([[0, 0, 1, 1, 0]]), label=3),
#                         Mask(np.array([[0, 0, 0, 0, 1]]), label=4),
#                     ],
#                 ),
#             ],
#             categories={
#                 AnnotationType.label: LabelCategories.from_iterable(
#                     ["background", "sky", "building", "person", "road"]
#                 ),
#                 AnnotationType.mask: MaskCategories(
#                     {
#                         0: (0, 0, 0),
#                         1: (0, 0, 64),
#                         2: (0, 128, 128),
#                         3: (128, 0, 64),
#                         4: (0, 192, 128),
#                     }
#                 ),
#             },
#         )

#         dataset = Dataset.import_from(DUMMY_DATASET_DIR_META_FILE, "synthia")

#         compare_datasets(self, expected_dataset, dataset, require_media=True)
