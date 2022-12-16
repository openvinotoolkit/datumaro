import os
import os.path as osp
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import numpy as np

import datumaro.plugins.data_formats.cityscapes as Cityscapes
from datumaro.components.annotation import AnnotationType, LabelCategories, Mask, MaskCategories
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.media import Image
from datumaro.plugins.data_formats.cityscapes import (
    TRAIN_CITYSCAPES_LABEL_MAP,
    CityscapesExporter,
    CityscapesImporter,
)
from datumaro.util.meta_file_util import parse_meta_file
from datumaro.util.test_utils import IGNORE_ALL, TestDir, check_save_and_load, compare_datasets

from ..requirements import Requirements, mark_requirement

from tests.utils.assets import get_test_asset_path

DUMMY_DATASET_DIR = get_test_asset_path("cityscapes_dataset", "dataset")
DUMMY_TRAIN_DATASET_DIR = get_test_asset_path("cityscapes_dataset", "train_dataset")


class CityscapesFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def test_can_write_and_parse_labelmap(self):
        src_label_map = Cityscapes.CITYSCAPES_LABEL_MAP

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, "label_colors.txt")

            Cityscapes.write_label_map(file_path, src_label_map)
            dst_label_map = Cityscapes.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_write_and_parse_dataset_meta_file(self):
        src_label_map = Cityscapes.CITYSCAPES_LABEL_MAP

        with TestDir() as test_dir:
            source_dataset = Dataset.from_iterable(
                [], categories=Cityscapes.make_cityscapes_categories(src_label_map)
            )

            CityscapesExporter.convert(source_dataset, test_dir, save_dataset_meta=True)
            dst_label_map = parse_meta_file(test_dir)

            self.assertEqual(src_label_map, dst_label_map)


class CityscapesImportTest(TestCase):
    @mark_requirement(Requirements.DATUM_267)
    def test_can_import(self):
        # is_crowd marks labels allowing to specify instance id
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000031",
                    subset="test",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=3, attributes={"is_crowd": True}),
                        Mask(
                            np.array([[0, 0, 1, 0, 0]]),
                            id=1,
                            label=27,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            np.array([[0, 0, 0, 1, 1]]),
                            id=2,
                            label=27,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000032",
                    subset="test",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            np.array([[1, 1, 0, 0, 0]]),
                            id=1,
                            label=31,
                            attributes={"is_crowd": False},
                        ),
                        Mask(np.array([[0, 0, 1, 0, 0]]), label=12, attributes={"is_crowd": True}),
                        Mask(np.array([[0, 0, 0, 1, 1]]), label=3, attributes={"is_crowd": True}),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000002_000045",
                    subset="train",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 1, 1]]), label=3, attributes={"is_crowd": True}),
                        Mask(
                            np.array([[0, 0, 1, 0, 0]]),
                            id=1,
                            label=24,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000019",
                    subset="val",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 1, 1]]), label=3, attributes={"is_crowd": True}),
                        Mask(
                            np.array([[0, 1, 1, 0, 0]]),
                            id=24,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                    ],
                ),
            ],
            categories=Cityscapes.make_cityscapes_categories(),
        )

        parsed_dataset = Dataset.import_from(DUMMY_DATASET_DIR, "cityscapes")

        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_import_with_train_label_map(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000031",
                    subset="test",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=19, attributes={"is_crowd": True}),
                        Mask(np.array([[0, 0, 1, 1, 1]]), label=14, attributes={"is_crowd": True}),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000032",
                    subset="test",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 0, 0]]), label=16, attributes={"is_crowd": True}),
                        Mask(np.array([[0, 0, 1, 0, 0]]), label=3, attributes={"is_crowd": True}),
                        Mask(np.array([[0, 0, 0, 1, 1]]), label=19, attributes={"is_crowd": True}),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000002_000045",
                    subset="train",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 0, 1, 1]]), label=19, attributes={"is_crowd": True}),
                        Mask(np.array([[0, 0, 1, 0, 0]]), label=11, attributes={"is_crowd": True}),
                    ],
                ),
                DatasetItem(
                    id="defaultcity/defaultcity_000001_000019",
                    subset="val",
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 1, 1, 1, 1]]), label=19, attributes={"is_crowd": True}),
                    ],
                ),
            ],
            categories=Cityscapes.make_cityscapes_categories(label_map=TRAIN_CITYSCAPES_LABEL_MAP),
        )

        parsed_dataset = Dataset.import_from(DUMMY_TRAIN_DATASET_DIR, "cityscapes")

        compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_267)
    def test_can_detect_cityscapes(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        self.assertEqual([CityscapesImporter.NAME], detected_formats)


class TestExtractorBase(DatasetBase):
    def _label(self, cityscapes_label):
        return self.categories()[AnnotationType.label].find(cityscapes_label)[0]

    def categories(self):
        return Cityscapes.make_cityscapes_categories()


class CityscapesExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="cityscapes",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="defaultcity_1_2",
                            subset="test",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[0, 0, 0, 1, 0]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                                Mask(
                                    np.array([[1, 0, 0, 0, 1]]),
                                    label=15,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                        DatasetItem(
                            id="defaultcity_3",
                            subset="val",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 1, 0, 1, 1]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 0, 1, 0, 0]]),
                                    label=5,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes", save_media=True),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="defaultcity_1_2",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 0, 0, 1, 0]]),
                                    label=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 1, 1, 0, 1]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                        DatasetItem(
                            id="defaultcity_1_3",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 1, 0, 1, 0]]),
                                    label=1,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 0, 1, 0, 1]]),
                                    label=2,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes", save_media=True),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="кириллица с пробелом",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 0, 0, 1, 1]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes", save_media=True),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_with_relative_path_in_id(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="test",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 0, 0, 1, 1]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 1, 1, 0, 0]]),
                                    label=24,
                                    id=1,
                                    attributes={"is_crowd": False},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes", save_media=True),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_with_no_masks(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="city_1_2",
                            subset="test",
                            media=Image(data=np.ones((2, 5, 3))),
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes", save_media=True),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_dataset_with_source_labelmap_undefined(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 1, 1]]), label=0),
                        Mask(np.array([[0, 1, 1, 0, 0]]), label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("a"),
                        ),
                        Mask(
                            np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("b"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = None
                label_map["a"] = None
                label_map["b"] = None
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(CityscapesExporter.convert, label_map="source", save_media=True),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_dataset_with_save_dataset_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(np.array([[1, 0, 0, 1, 1]]), label=0),
                        Mask(np.array([[0, 1, 1, 0, 0]]), label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("a"),
                        ),
                        Mask(
                            np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("b"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = None
                label_map["a"] = None
                label_map["b"] = None
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(
                    CityscapesExporter.convert,
                    label_map="source",
                    save_media=True,
                    save_dataset_meta=True,
                ),
                test_dir,
                target_dataset=DstExtractor(),
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))

    @mark_requirement(Requirements.DATUM_267)
    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            np.array([[1, 0, 0, 1, 1]]),
                            label=1,
                            id=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            np.array([[0, 1, 1, 0, 0]]),
                            label=2,
                            id=2,
                            attributes={"is_crowd": False},
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Cityscapes.make_cityscapes_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    media=Image(data=np.ones((1, 5, 3))),
                    annotations=[
                        Mask(
                            np.array([[1, 0, 0, 1, 1]]),
                            attributes={"is_crowd": False},
                            id=1,
                            label=self._label("label_1"),
                        ),
                        Mask(
                            np.array([[0, 1, 1, 0, 0]]),
                            attributes={"is_crowd": False},
                            id=2,
                            label=self._label("label_2"),
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = (0, 0, 0)
                label_map["label_1"] = (1, 2, 3)
                label_map["label_2"] = (3, 2, 1)
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(CityscapesExporter.convert, label_map="source", save_media=True),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_267)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="q", media=Image(path="q.JPEG", data=np.zeros((4, 3, 3)))),
                        DatasetItem(
                            id="w",
                            media=Image(path="w.bmp", data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[1, 0, 0, 1, 0]]),
                                    label=0,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[0, 1, 1, 0, 1]]),
                                    label=1,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["a"] = None
                label_map["b"] = None
                return Cityscapes.make_cityscapes_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, save_media=True),
                test_dir,
                require_media=True,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self):
        src_mask_cat = MaskCategories.generate(2, include_background=False)

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image(data=np.ones((2, 1, 3))),
                    annotations=[Mask(np.ones((2, 1)), label=2, id=1)],
                ),
                DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image(data=np.ones((2, 2, 3))),
                    annotations=[Mask(np.ones((2, 2)), label=1, id=1)],
                ),
            ],
            categories=Cityscapes.make_cityscapes_categories(
                OrderedDict(
                    [
                        ("a", src_mask_cat.colormap[0]),
                        ("b", src_mask_cat.colormap[1]),
                    ]
                )
            ),
        )

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        1,
                        subset="a",
                        media=Image(data=np.ones((2, 1, 3))),
                        annotations=[Mask(np.ones((2, 1)), label=1)],
                    ),
                    DatasetItem(
                        2,
                        subset="b",
                        media=Image(data=np.ones((2, 2, 3))),
                        annotations=[Mask(np.ones((2, 2)), label=0)],
                    ),
                    DatasetItem(
                        3,
                        subset="c",
                        media=Image(data=np.ones((2, 3, 3))),
                        annotations=[Mask(np.ones((2, 2)), label=0)],
                    ),
                ],
                categories={
                    AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                    AnnotationType.mask: src_mask_cat,
                },
            )
            dataset.export(path, "cityscapes", save_media=True)

            dataset.put(DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a", "b"}, set(os.listdir(osp.join(path, "gtFine"))))
            self.assertEqual(
                {"1_gtFine_color.png", "1_gtFine_instanceIds.png", "1_gtFine_labelIds.png"},
                set(os.listdir(osp.join(path, "gtFine", "a"))),
            )
            self.assertEqual(
                {"2_gtFine_color.png", "2_gtFine_instanceIds.png", "2_gtFine_labelIds.png"},
                set(os.listdir(osp.join(path, "gtFine", "b"))),
            )
            self.assertEqual({"a", "b"}, set(os.listdir(osp.join(path, "imgsFine", "leftImg8bit"))))
            self.assertEqual(
                {"1_leftImg8bit.png", "2_leftImg8bit.png"},
                set(os.listdir(osp.join(path, "imgsFine", "leftImg8bit", "a"))),
            )
            self.assertEqual(
                {"2_leftImg8bit.png"},
                set(os.listdir(osp.join(path, "imgsFine", "leftImg8bit", "b"))),
            )
            compare_datasets(
                self,
                expected,
                Dataset.import_from(path, "cityscapes"),
                require_media=True,
                ignored_attrs=IGNORE_ALL,
            )

    @mark_requirement(Requirements.DATUM_BUG_470)
    def test_can_save_and_load_without_image_saving(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a",
                            subset="test",
                            media=Image(data=np.ones((1, 5, 3))),
                            annotations=[
                                Mask(
                                    np.array([[0, 1, 1, 1, 0]]),
                                    label=3,
                                    attributes={"is_crowd": True},
                                ),
                                Mask(
                                    np.array([[1, 0, 0, 0, 1]]),
                                    label=4,
                                    attributes={"is_crowd": True},
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(CityscapesExporter.convert, label_map="cityscapes"),
                test_dir,
            )
