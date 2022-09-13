import os.path as osp
import shutil
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    LabelCategories,
    Mask,
    MaskCategories,
    Polygon,
)
from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.mapillary_vistas_format.format import (
    MapillaryVistasLabelMaps,
    make_mapillary_instance_categories,
)
from datumaro.util.test_utils import TestDir, compare_datasets

from .requirements import Requirements, mark_requirement

DUMMY_DATASET_V1_2 = osp.join(osp.dirname(__file__), "assets", "mapillary_vistas_dataset", "v1.2")
DUMMY_DATASET_V2_0 = osp.join(osp.dirname(__file__), "assets", "mapillary_vistas_dataset", "v2.0")
DUMMY_DATASET_WITH_META_FILE = osp.join(
    osp.dirname(__file__), "assets", "mapillary_vistas_dataset", "dataset_with_meta_file"
)


class MapillaryVistasImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v1_2(self):
        label_cat = LabelCategories.from_iterable(
            [
                ("animal--bird", "animal"),
                ("construction--barrier--curb", "construction"),
                ("human--person", "human"),
            ]
        )
        mask_cat = MaskCategories({0: (10, 50, 90), 1: (20, 30, 80), 2: (30, 70, 40)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="val",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), label=0),
                        Mask(image=np.array([[0, 0, 1, 1, 0]] * 5), label=1),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="1",
                    subset="train",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), label=0, id=0),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), label=0, id=1),
                        Mask(image=np.array([[0, 0, 1, 1, 0]] * 5), label=1, id=0),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]] * 5), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_V1_2, "mapillary_vistas")

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_original_config(self):
        exptected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="val",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), label=0),
                        Mask(image=np.array([[0, 0, 1, 1, 0]] * 5), label=1),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories=make_mapillary_instance_categories(MapillaryVistasLabelMaps["v1.2"]),
        )

        imported_dataset = Dataset.import_from(
            osp.join(DUMMY_DATASET_V1_2, "val"), "mapillary_vistas", use_original_config=True
        )

        compare_datasets(self, exptected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v1_2_wo_images(self):
        exptected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="dataset",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), label=0),
                        Mask(image=np.array([[0, 0, 1, 1, 0]] * 5), label=1),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), label=2),
                    ],
                ),
            ],
            categories=make_mapillary_instance_categories(MapillaryVistasLabelMaps["v1.2"]),
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(osp.join(DUMMY_DATASET_V1_2, "val"), dataset_path)
            shutil.rmtree(osp.join(dataset_path, "images"))

            imported_dataset = Dataset.import_from(
                dataset_path, "mapillary_vistas", use_original_config=True
            )

            compare_datasets(self, exptected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v2_0_instances(self):
        label_cat = LabelCategories.from_iterable(
            [
                ("animal--bird", "animal"),
                ("construction--barrier--separator", "construction"),
                ("object--vehicle--bicycle", "object"),
            ]
        )

        mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="val",
                    annotations=[
                        Mask(image=np.array([[1, 1, 1, 0, 0]] * 5), id=0, label=0),
                        Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=0, label=1),
                        Polygon(points=[0, 0, 1, 0, 2, 0, 2, 4, 0, 4], label=0),
                        Polygon(points=[3, 0, 4, 0, 4, 1, 4, 4, 3, 4], label=1),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="1",
                    subset="val",
                    annotations=[
                        Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=0, label=1),
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), id=0, label=2),
                        Mask(image=np.array([[0, 0, 0, 1, 1]] * 5), id=1, label=2),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=1),
                        Polygon(points=[0, 0, 1, 0, 1, 4, 4, 0, 0, 0], label=2),
                        Polygon(points=[3, 0, 4, 0, 4, 4, 3, 4, 3, 0], label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    annotations=[
                        Mask(image=np.array([[1, 0, 0, 0, 0]] * 5), id=0, label=0),
                        Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), id=1, label=0),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), id=2, label=0),
                        Mask(image=np.array([[0, 1, 0, 0, 0]] * 5), id=0, label=1),
                        Mask(image=np.array([[0, 0, 0, 1, 0]] * 5), id=1, label=1),
                        Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=0),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=0),
                        Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=0),
                        Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=1),
                        Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=1),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_V2_0, "mapillary_vistas_instances")

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v2_0_panoptic(self):
        label_cat = LabelCategories.from_iterable(
            [
                ("animal--bird", "animal"),
                ("construction--barrier--separator", "construction"),
                ("object--vehicle--bicycle", "object"),
            ]
        )

        mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="val",
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 1, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=0,
                            attributes={"is_crowd": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            id=2,
                            group=2,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(points=[0, 0, 1, 0, 2, 0, 2, 4, 0, 4], label=0),
                        Polygon(points=[3, 0, 4, 0, 4, 1, 4, 4, 3, 4], label=1),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="1",
                    subset="val",
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 0, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=2,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]] * 5),
                            id=2,
                            group=2,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            id=3,
                            group=3,
                            label=2,
                            attributes={"is_crowd": True},
                        ),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=1),
                        Polygon(points=[0, 0, 1, 0, 1, 4, 4, 0, 0, 0], label=2),
                        Polygon(points=[3, 0, 4, 0, 4, 4, 3, 4, 3, 0], label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            id=2,
                            group=2,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]] * 5),
                            id=3,
                            group=3,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 0]] * 5),
                            id=4,
                            group=4,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 0, 1]] * 5),
                            id=5,
                            group=5,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=0),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=0),
                        Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=0),
                        Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=1),
                        Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=1),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_V2_0, "mapillary_vistas_panoptic")

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v2_0_panoptic_with_keeping_category_ids(self):
        labels = [f"class-{i}" for i in range(101)]
        labels[1] = ("animal--bird", "animal")
        labels[10] = ("construction--barrier--separator", "construction")
        labels[100] = ("object--vehicle--bicycle", "object")

        label_cat = LabelCategories.from_iterable(labels)
        mask_cat = MaskCategories({1: (165, 42, 42), 10: (128, 128, 128), 100: (119, 11, 32)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0",
                    subset="val",
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 1, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=1,
                            attributes={"is_crowd": True},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            id=2,
                            group=2,
                            label=10,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(points=[0, 0, 1, 0, 2, 0, 2, 4, 0, 4], label=1),
                        Polygon(points=[3, 0, 4, 0, 4, 1, 4, 4, 3, 4], label=10),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="1",
                    subset="val",
                    annotations=[
                        Mask(
                            image=np.array([[1, 1, 0, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=100,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]] * 5),
                            id=2,
                            group=2,
                            label=10,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 1]] * 5),
                            id=3,
                            group=3,
                            label=100,
                            attributes={"is_crowd": True},
                        ),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=10),
                        Polygon(points=[0, 0, 1, 0, 1, 4, 4, 0, 0, 0], label=100),
                        Polygon(points=[3, 0, 4, 0, 4, 4, 3, 4, 3, 0], label=100),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            id=2,
                            group=2,
                            label=10,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]] * 5),
                            id=3,
                            group=3,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 0]] * 5),
                            id=4,
                            group=4,
                            label=10,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 0, 1]] * 5),
                            id=5,
                            group=5,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=1),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=1),
                        Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=1),
                        Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=10),
                        Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=10),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        imported_dataset = Dataset.import_from(
            DUMMY_DATASET_V2_0, "mapillary_vistas_panoptic", keep_original_category_ids=True
        )

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_v2_0_panoptic_wo_images(self):
        label_cat = LabelCategories.from_iterable(
            [
                ("animal--bird", "animal"),
                ("construction--barrier--separator", "construction"),
                ("object--vehicle--bicycle", "object"),
            ]
        )

        mask_cat = MaskCategories({0: (165, 42, 42), 1: (128, 128, 128), 2: (119, 11, 32)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2",
                    subset="dataset",
                    annotations=[
                        Mask(
                            image=np.array([[1, 0, 0, 0, 0]] * 5),
                            id=1,
                            group=1,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 1, 0, 0, 0]] * 5),
                            id=2,
                            group=2,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 1, 0, 0]] * 5),
                            id=3,
                            group=3,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 1, 0]] * 5),
                            id=4,
                            group=4,
                            label=1,
                            attributes={"is_crowd": False},
                        ),
                        Mask(
                            image=np.array([[0, 0, 0, 0, 1]] * 5),
                            id=5,
                            group=5,
                            label=0,
                            attributes={"is_crowd": False},
                        ),
                        Polygon(points=[0, 0, 0, 1, 0, 2, 0, 3, 0, 4], label=0),
                        Polygon(points=[2, 0, 2, 1, 2, 2, 2, 3, 2, 4], label=0),
                        Polygon(points=[4, 0, 4, 1, 4, 2, 4, 3, 4, 4], label=0),
                        Polygon(points=[1, 0, 1, 1, 1, 2, 1, 3, 1, 4], label=1),
                        Polygon(points=[3, 0, 3, 1, 3, 2, 3, 3, 3, 4], label=1),
                    ],
                )
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(osp.join(DUMMY_DATASET_V2_0, "train"), dataset_path)
            shutil.rmtree(osp.join(dataset_path, "images"))

            imported_dataset = Dataset.import_from(dataset_path, "mapillary_vistas_panoptic")

            compare_datasets(self, expected_dataset, imported_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_meta_file(self):
        label_cat = LabelCategories.from_iterable(
            [
                ("animal--bird", "animal"),
                ("construction--barrier--curb", "construction"),
                ("human--person", "human"),
            ]
        )
        mask_cat = MaskCategories({0: (10, 50, 90), 1: (20, 30, 80), 2: (30, 70, 40)})

        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 0, 0]] * 5), label=0, id=0),
                        Mask(image=np.array([[0, 0, 0, 0, 1]] * 5), label=0, id=1),
                        Mask(image=np.array([[0, 0, 1, 1, 0]] * 5), label=1, id=0),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
                DatasetItem(
                    id="2",
                    subset="train",
                    annotations=[
                        Mask(image=np.array([[1, 1, 0, 1, 1]] * 5), label=1),
                        Mask(image=np.array([[0, 0, 1, 0, 0]] * 5), label=2),
                    ],
                    media=Image(data=np.ones((5, 5, 3))),
                ),
            ],
            categories={AnnotationType.label: label_cat, AnnotationType.mask: mask_cat},
        )

        imported_dataset = Dataset.import_from(DUMMY_DATASET_WITH_META_FILE, "mapillary_vistas")

        compare_datasets(self, expected_dataset, imported_dataset, require_media=True)
