import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
from collections import OrderedDict
from functools import partial
from unittest import TestCase

import numpy as np
from lxml import etree as ElementTree  # nosec

import datumaro.plugins.data_formats.voc.format as VOC
from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
)
from datumaro.components.dataset import Dataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AnnotationImportError,
    InvalidAnnotationError,
    InvalidFieldError,
    ItemImportError,
    MissingFieldError,
    UndeclaredLabelError,
)
from datumaro.components.media import Image
from datumaro.plugins.data_formats.voc.exporter import (
    VocActionExporter,
    VocClassificationExporter,
    VocDetectionExporter,
    VocExporter,
    VocLayoutExporter,
    VocSegmentationExporter,
)
from datumaro.plugins.data_formats.voc.importer import VocImporter
from datumaro.util.image import save_image
from datumaro.util.mask_tools import load_mask
from datumaro.util.test_utils import (
    TestDir,
    check_save_and_load,
    compare_datasets,
    compare_datasets_strict,
)

from .requirements import Requirements, mark_requirement


class VocFormatTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_colormap_generator(self):
        reference = np.array(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
                [224, 224, 192],  # ignored
            ]
        )

        self.assertTrue(np.array_equal(reference, list(VOC.VocColormap.values())))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_write_and_parse_labelmap(self):
        src_label_map = VOC.make_voc_label_map()
        src_label_map["qq"] = [None, ["part1", "part2"], ["act1", "act2"]]
        src_label_map["ww"] = [(10, 20, 30), [], ["act3"]]

        with TestDir() as test_dir:
            file_path = osp.join(test_dir, "test.txt")

            VOC.write_label_map(file_path, src_label_map)
            dst_label_map = VOC.parse_label_map(file_path)

            self.assertEqual(src_label_map, dst_label_map)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_write_and_parse_dataset_meta_file(self):
        src_label_map = VOC.make_voc_label_map()
        src_label_map["qq"] = [None, ["part1", "part2"], ["act1", "act2"]]
        src_label_map["ww"] = [(10, 20, 30), [], ["act3"]]

        with TestDir() as test_dir:
            VOC.write_meta_file(test_dir, src_label_map)
            dst_label_map = VOC.parse_meta_file(test_dir)

            self.assertEqual(src_label_map, dst_label_map)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_invalid_line_in_labelmap(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "labelmap.txt")
            with open(path, "w") as f:
                f.write("a\n")

            with self.assertRaisesRegex(InvalidAnnotationError, "Expected 4 ':'-separated fields"):
                VOC.parse_label_map(path)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_repeated_label_in_labelmap(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "labelmap.txt")
            with open(path, "w") as f:
                f.write("a:::\n")
                f.write("a:::\n")

            with self.assertRaisesRegex(InvalidAnnotationError, "already defined"):
                VOC.parse_label_map(path)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_invalid_color_in_labelmap(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "labelmap.txt")
            with open(path, "w") as f:
                f.write("a:10,20::\n")

            with self.assertRaisesRegex(InvalidAnnotationError, "Expected an 'r,g,b' triplet"):
                VOC.parse_label_map(path)


class TestExtractorBase(DatasetBase):
    def _label(self, voc_label):
        return self.categories()[AnnotationType.label].find(voc_label)[0]

    def categories(self):
        return VOC.make_voc_categories()


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "voc_dataset", "voc_dataset1")
DUMMY_DATASET2_DIR = osp.join(osp.dirname(__file__), "assets", "voc_dataset", "voc_dataset2")
DUMMY_DATASET3_DIR = osp.join(osp.dirname(__file__), "assets", "voc_dataset", "voc_dataset3")


class VocImportTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="2007_000001",
                            subset="train",
                            media=Image(data=np.ones((10, 20, 3))),
                            annotations=[
                                Label(self._label(l.name)) for l in VOC.VocLabel if l.value % 2 == 1
                            ]
                            + [
                                Bbox(
                                    1,
                                    2,
                                    2,
                                    2,
                                    label=self._label("cat"),
                                    attributes={
                                        "pose": VOC.VocPose(1).name,
                                        "truncated": True,
                                        "difficult": False,
                                        "occluded": False,
                                    },
                                    id=1,
                                    group=1,
                                ),
                                # Only main boxes denote instances (have ids)
                                Mask(
                                    image=np.ones([10, 20]),
                                    label=self._label(VOC.VocLabel(2).name),
                                    group=1,
                                ),
                                Bbox(
                                    4,
                                    5,
                                    2,
                                    2,
                                    label=self._label("person"),
                                    attributes={
                                        "truncated": False,
                                        "difficult": False,
                                        "occluded": False,
                                        **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                                    },
                                    id=2,
                                    group=2,
                                ),
                                # Only main boxes denote instances (have ids)
                                Bbox(
                                    5.5,
                                    6,
                                    2,
                                    2,
                                    label=self._label(VOC.VocBodyPart(1).name),
                                    group=2,
                                ),
                            ],
                        ),
                        DatasetItem(
                            id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                        ),
                    ]
                )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "voc")

        compare_datasets(self, DstExtractor(), dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_voc_classification_dataset(self):
        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="2007_000001",
                            subset="train",
                            media=Image(data=np.ones((10, 20, 3))),
                            annotations=[
                                Label(self._label(l.name)) for l in VOC.VocLabel if l.value % 2 == 1
                            ],
                        ),
                        DatasetItem(
                            id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                        ),
                    ]
                )

        expected_dataset = DstExtractor()

        rpath = osp.join("ImageSets", "Main", "train.txt")
        matrix = [
            ("voc_classification", "", ""),
            ("voc_classification", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_voc_layout_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=15,
                            id=2,
                            group=2,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                                **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                            },
                        ),
                        Bbox(5.5, 6.0, 2.0, 2.0, label=22, group=2),
                    ],
                ),
                DatasetItem(
                    id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                ),
            ],
            categories=VOC.make_voc_categories(),
        )

        rpath = osp.join("ImageSets", "Layout", "train.txt")
        matrix = [
            ("voc_layout", "", ""),
            ("voc_layout", "train", rpath),
            ("voc", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_voc_detection_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            1.0,
                            2.0,
                            2.0,
                            2.0,
                            label=8,
                            id=1,
                            group=1,
                            attributes={
                                "difficult": False,
                                "truncated": True,
                                "occluded": False,
                                "pose": "Unspecified",
                            },
                        ),
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=15,
                            id=2,
                            group=2,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                                **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                ),
            ],
            categories=VOC.make_voc_categories(),
        )

        rpath = osp.join("ImageSets", "Main", "train.txt")
        matrix = [
            ("voc_detection", "", ""),
            ("voc_detection", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_voc_segmentation_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image(data=np.ones((10, 20, 3))),
                    annotations=[Mask(image=np.ones([10, 20]), label=2, group=1)],
                ),
                DatasetItem(
                    id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                ),
            ],
            categories=VOC.make_voc_categories(),
        )

        rpath = osp.join("ImageSets", "Segmentation", "train.txt")
        matrix = [
            ("voc_segmentation", "", ""),
            ("voc_segmentation", "train", rpath),
            ("voc", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_voc_action_dataset(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            4.0,
                            5.0,
                            2.0,
                            2.0,
                            label=15,
                            id=2,
                            group=2,
                            attributes={
                                "difficult": False,
                                "truncated": False,
                                "occluded": False,
                                **{a.name: a.value % 2 == 1 for a in VOC.VocAction},
                            },
                        )
                    ],
                ),
                DatasetItem(
                    id="2007_000002", subset="test", media=Image(data=np.ones((10, 20, 3)))
                ),
            ],
            categories=VOC.make_voc_categories(),
        )

        rpath = osp.join("ImageSets", "Action", "train.txt")
        matrix = [
            ("voc_action", "", ""),
            ("voc_action", "train", rpath),
            ("voc", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect_voc(self):
        env = Environment()

        for path in [DUMMY_DATASET_DIR, DUMMY_DATASET2_DIR]:
            with self.subTest(path=path):
                detected_formats = env.detect_dataset(path)
                self.assertEqual([VocImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_BUG_583)
    def test_can_import_voc_dataset_with_empty_lines_in_subset_lists(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="2007_000001",
                    subset="train",
                    media=Image(data=np.ones((10, 20, 3))),
                    annotations=[
                        Bbox(
                            1.0,
                            2.0,
                            2.0,
                            2.0,
                            label=8,
                            id=1,
                            group=1,
                            attributes={
                                "difficult": False,
                                "truncated": True,
                                "occluded": False,
                                "pose": "Unspecified",
                            },
                        )
                    ],
                )
            ],
            categories=VOC.make_voc_categories(),
        )

        rpath = osp.join("ImageSets", "Main", "train.txt")
        matrix = [
            ("voc_detection", "", ""),
            ("voc_detection", "train", rpath),
        ]
        for format, subset, path in matrix:
            with self.subTest(format=format, subset=subset, path=path):
                if subset:
                    expected = expected_dataset.get_subset(subset)
                else:
                    expected = expected_dataset

                actual = Dataset.import_from(osp.join(DUMMY_DATASET3_DIR, path), format)

                compare_datasets(self, expected, actual, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        formats = [
            "voc",
            "voc_classification",
            "voc_detection",
            "voc_action",
            "voc_layout",
            "voc_segmentation",
        ]

        for fmt in formats:
            with self.subTest(fmt=fmt):
                source = Dataset.import_from(DUMMY_DATASET_DIR, format=fmt)

                parsed = pickle.loads(pickle.dumps(source))  # nosec

                compare_datasets_strict(self, source, parsed)


class VocExtractorTest(TestCase):
    # ?xml... must be in the file beginning
    XML_ANNOTATION_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
<filename>a.jpg</filename>
<size><width>20</width><height>10</height><depth>3</depth></size>
<object>
    <name>person</name>
    <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox>
    <difficult>1</difficult>
    <truncated>1</truncated>
    <occluded>1</occluded>
    <point><x>1</x><y>1</y></point>
    <attributes><attribute><name>a</name><value>42</value></attribute></attributes>
    <actions><jumping>1</jumping></actions>
    <part>
        <name>head</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3</xmax><ymax>4</ymax></bndbox>
    </part>
</object>
</annotation>
    """

    @classmethod
    def _write_xml_dataset(cls, root_dir, fmt_dir="Main", mangle_xml=None):
        subset_file = osp.join(root_dir, "ImageSets", fmt_dir, "test.txt")
        os.makedirs(osp.dirname(subset_file))
        with open(subset_file, "w") as f:
            f.write("a\n" if fmt_dir != "Layout" else "a 0\n")

        ann_file = osp.join(root_dir, "Annotations", "a.xml")
        os.makedirs(osp.dirname(ann_file))
        with open(ann_file, "wb") as f:
            xml = ElementTree.fromstring(cls.XML_ANNOTATION_TEMPLATE.encode())
            if mangle_xml:
                mangle_xml(xml)
            f.write(ElementTree.tostring(xml))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_parse_xml_without_errors(self):
        formats = [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ]

        for fmt, fmt_dir in formats:
            with self.subTest(fmt=fmt):
                with TestDir() as test_dir:
                    self._write_xml_dataset(test_dir, fmt_dir=fmt_dir)

                    dataset = Dataset.import_from(test_dir, fmt)
                    dataset.init_cache()
                    self.assertEqual(len(dataset), 1)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_quotes_in_lists_of_layout_task(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Layout", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write('"qwe 1\n')

            with self.assertRaisesRegex(
                InvalidAnnotationError, "unexpected number of quotes in filename"
            ):
                Dataset.import_from(test_dir, format="voc_layout")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label_in_xml(self):
        formats = [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ]

        for fmt, fmt_dir in formats:
            with self.subTest(fmt=fmt):
                with TestDir() as test_dir:

                    def mangle_xml(xml: ElementTree.ElementBase):
                        xml.find("object/name").text = "test"

                    self._write_xml_dataset(test_dir, fmt_dir=fmt_dir, mangle_xml=mangle_xml)

                    with self.assertRaises(AnnotationImportError) as capture:
                        Dataset.import_from(test_dir, format=fmt).init_cache()
                    self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
                    self.assertEqual(capture.exception.__cause__.id, "test")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_field_in_xml(self):
        formats = [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ]

        for fmt, fmt_dir in formats:
            with self.subTest(fmt=fmt):
                for key in [
                    "object/name",
                    "object/bndbox",
                    "object/bndbox/xmin",
                    "object/bndbox/ymin",
                    "object/bndbox/xmax",
                    "object/bndbox/ymax",
                    "object/part/name",
                    "object/part/bndbox/xmin",
                    "object/part/bndbox/ymin",
                    "object/part/bndbox/xmax",
                    "object/part/bndbox/ymax",
                    "object/point/x",
                    "object/point/y",
                    "object/attributes/attribute/name",
                    "object/attributes/attribute/value",
                ]:
                    with self.subTest(key=key):
                        with TestDir() as test_dir:

                            def mangle_xml(xml: ElementTree.ElementBase):
                                for elem in xml.findall(key):
                                    elem.getparent().remove(elem)

                            self._write_xml_dataset(
                                test_dir, fmt_dir=fmt_dir, mangle_xml=mangle_xml
                            )

                            with self.assertRaises(ItemImportError) as capture:
                                Dataset.import_from(test_dir, format=fmt).init_cache()
                            self.assertIsInstance(capture.exception.__cause__, MissingFieldError)
                            self.assertIn(osp.basename(key), capture.exception.__cause__.name)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_field_in_xml(self):
        formats = [
            ("voc_detection", "Main"),
            ("voc_layout", "Layout"),
            ("voc_action", "Action"),
        ]

        for fmt, fmt_dir in formats:
            with self.subTest(fmt=fmt):
                for key, value in [
                    ("object/bndbox/xmin", "a"),
                    ("object/bndbox/ymin", "a"),
                    ("object/bndbox/xmax", "a"),
                    ("object/bndbox/ymax", "a"),
                    ("object/part/bndbox/xmin", "a"),
                    ("object/part/bndbox/ymin", "a"),
                    ("object/part/bndbox/xmax", "a"),
                    ("object/part/bndbox/ymax", "a"),
                    ("size/width", "a"),
                    ("size/height", "a"),
                    ("object/occluded", "a"),
                    ("object/difficult", "a"),
                    ("object/truncated", "a"),
                    ("object/point/x", "a"),
                    ("object/point/y", "a"),
                    ("object/actions/jumping", "a"),
                ]:
                    with self.subTest(key=key):
                        with TestDir() as test_dir:

                            def mangle_xml(xml: ElementTree.ElementBase):
                                xml.find(key).text = value

                            self._write_xml_dataset(
                                test_dir, fmt_dir=fmt_dir, mangle_xml=mangle_xml
                            )

                            with self.assertRaises(ItemImportError) as capture:
                                Dataset.import_from(test_dir, format=fmt).init_cache()
                            self.assertIsInstance(capture.exception.__cause__, InvalidFieldError)
                            self.assertIn(osp.basename(key), capture.exception.__cause__.name)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_field_in_classification(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write("a\n")

            ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
            with open(ann_file, "w") as f:
                f.write("a\n")

            with self.assertRaisesRegex(InvalidAnnotationError, "invalid number of fields"):
                Dataset.import_from(test_dir, format="voc_classification").init_cache()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_parse_classification_without_errors(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write("a\n")
                f.write("b\n")
                f.write("c\n")

            ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
            with open(ann_file, "w") as f:
                f.write("a -1\n")
                f.write("b 0\n")
                f.write("c 1\n")

            parsed = Dataset.import_from(test_dir, format="voc_classification")

            expected = Dataset.from_iterable(
                [
                    DatasetItem("a", subset="test"),
                    DatasetItem("b", subset="test"),
                    DatasetItem("c", subset="test", annotations=[Label(VOC.VocLabel.cat.value)]),
                ],
                categories=VOC.make_voc_categories(),
            )
            compare_datasets(self, expected, parsed)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_annotation_value_in_classification(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Main", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write("a\n")

            ann_file = osp.join(test_dir, "ImageSets", "Main", "cat_test.txt")
            with open(ann_file, "w") as f:
                f.write("a 3\n")

            with self.assertRaisesRegex(InvalidAnnotationError, "unexpected class existence value"):
                Dataset.import_from(test_dir, format="voc_classification").init_cache()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label_in_segmentation_cls_mask(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Segmentation", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write("a\n")

            ann_file = osp.join(test_dir, "SegmentationClass", "a.png")
            os.makedirs(osp.dirname(ann_file))
            save_image(ann_file, np.array([[30]], dtype=np.uint8))

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, format="voc_segmentation").init_cache()
            self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
            self.assertEqual(capture.exception.__cause__.id, "30")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label_in_segmentation_both_masks(self):
        with TestDir() as test_dir:
            subset_file = osp.join(test_dir, "ImageSets", "Segmentation", "test.txt")
            os.makedirs(osp.dirname(subset_file))
            with open(subset_file, "w") as f:
                f.write("a\n")

            cls_file = osp.join(test_dir, "SegmentationClass", "a.png")
            os.makedirs(osp.dirname(cls_file))
            save_image(cls_file, np.array([[30]], dtype=np.uint8))

            inst_file = osp.join(test_dir, "SegmentationObject", "a.png")
            os.makedirs(osp.dirname(inst_file))
            save_image(inst_file, np.array([[1]], dtype=np.uint8))

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, format="voc_segmentation").init_cache()
            self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
            self.assertEqual(capture.exception.__cause__.id, "30")


class VocExporterTest(TestCase):
    def _test_save_and_load(
        self, source_dataset, converter, test_dir, target_dataset=None, importer_args=None, **kwargs
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="voc",
            target_dataset=target_dataset,
            importer_args=importer_args,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_cls(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/0",
                            subset="a",
                            annotations=[
                                Label(1),
                                Label(2),
                                Label(3),
                            ],
                        ),
                        DatasetItem(
                            id=1,
                            subset="b",
                            annotations=[
                                Label(4),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocClassificationExporter.convert, label_map="voc"),
                test_dir,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_det(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/1",
                            subset="a",
                            annotations=[
                                Bbox(2, 3, 4, 5, label=2, attributes={"occluded": True}),
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=3,
                                    attributes={"truncated": True},
                                ),
                            ],
                        ),
                        DatasetItem(
                            id=2,
                            subset="b",
                            annotations=[
                                Bbox(
                                    5,
                                    4,
                                    6,
                                    5,
                                    label=3,
                                    attributes={"difficult": True},
                                ),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/1",
                            subset="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    id=1,
                                    group=1,
                                    attributes={
                                        "truncated": False,
                                        "difficult": False,
                                        "occluded": True,
                                    },
                                ),
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=3,
                                    id=2,
                                    group=2,
                                    attributes={
                                        "truncated": True,
                                        "difficult": False,
                                        "occluded": False,
                                    },
                                ),
                            ],
                        ),
                        DatasetItem(
                            id=2,
                            subset="b",
                            annotations=[
                                Bbox(
                                    5,
                                    4,
                                    6,
                                    5,
                                    label=3,
                                    id=1,
                                    group=1,
                                    attributes={
                                        "truncated": False,
                                        "difficult": True,
                                        "occluded": False,
                                    },
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocDetectionExporter.convert, label_map="voc"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_segm(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="a",
                            annotations=[
                                # overlapping masks, the first should be truncated
                                # the second and third are different instances
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, z_order=3),
                                Mask(image=np.array([[0, 1, 1, 1, 0]]), label=4, z_order=1),
                                Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, z_order=2),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="a",
                            annotations=[
                                Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4, group=1),
                                Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, group=2),
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, group=3),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocSegmentationExporter.convert, label_map="voc"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_segm_unpainted(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            annotations=[
                                # overlapping masks, the first should be truncated
                                # the second and third are different instances
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, z_order=3),
                                Mask(image=np.array([[0, 1, 1, 1, 0]]), label=4, z_order=1),
                                Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, z_order=2),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            annotations=[
                                Mask(image=np.array([[0, 0, 1, 0, 0]]), label=4, group=1),
                                Mask(image=np.array([[1, 1, 0, 0, 0]]), label=3, group=2),
                                Mask(image=np.array([[0, 0, 0, 1, 0]]), label=3, group=3),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocSegmentationExporter.convert, label_map="voc", apply_colormap=False),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_segm_with_many_instances(self):
        def bit(x, y, shape):
            mask = np.zeros(shape)
            mask[y, x] = 1
            return mask

        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            annotations=[
                                Mask(
                                    image=bit(x, y, shape=[10, 10]),
                                    label=self._label(VOC.VocLabel(3).name),
                                    z_order=10 * y + x + 1,
                                )
                                for y in range(10)
                                for x in range(10)
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id=1,
                            subset="a",
                            annotations=[
                                Mask(
                                    image=bit(x, y, shape=[10, 10]),
                                    label=self._label(VOC.VocLabel(3).name),
                                    group=10 * y + x + 1,
                                )
                                for y in range(10)
                                for x in range(10)
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocSegmentationExporter.convert, label_map="voc"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_layout(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    id=1,
                                    group=1,
                                    attributes={
                                        "pose": VOC.VocPose(1).name,
                                        "truncated": True,
                                        "difficult": False,
                                        "occluded": False,
                                    },
                                ),
                                Bbox(
                                    2, 3, 1, 1, label=self._label(VOC.VocBodyPart(1).name), group=1
                                ),
                                Bbox(
                                    5, 4, 3, 2, label=self._label(VOC.VocBodyPart(2).name), group=1
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(), partial(VocLayoutExporter.convert, label_map="voc"), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_voc_action(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    attributes={
                                        "truncated": True,
                                        VOC.VocAction(1).name: True,
                                        VOC.VocAction(2).name: True,
                                    },
                                ),
                                Bbox(
                                    5,
                                    4,
                                    3,
                                    2,
                                    label=self._label("person"),
                                    attributes={
                                        "truncated": True,
                                        VOC.VocAction(1).name: True,
                                        VOC.VocAction(2).name: True,
                                    },
                                ),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a/b/1",
                            subset="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    id=1,
                                    group=1,
                                    attributes={
                                        "truncated": True,
                                        "difficult": False,
                                        "occluded": False,
                                        # no attributes here in the label categories
                                    },
                                ),
                                Bbox(
                                    5,
                                    4,
                                    3,
                                    2,
                                    label=self._label("person"),
                                    id=2,
                                    group=2,
                                    attributes={
                                        "truncated": True,
                                        "difficult": False,
                                        "occluded": False,
                                        VOC.VocAction(1).name: True,
                                        VOC.VocAction(2).name: True,
                                        **{
                                            a.name: False
                                            for a in VOC.VocAction
                                            if a.value not in {1, 2}
                                        },
                                    },
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocActionExporter.convert, label_map="voc", allow_attributes=False),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_subsets(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id=1),
                        DatasetItem(id=2),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", tasks=task),
                    test_dir,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="кириллица с пробелом 1"),
                        DatasetItem(
                            id="кириллица с пробелом 2", media=Image(data=np.ones([4, 5, 3]))
                        ),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", tasks=task, save_media=True),
                    test_dir,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_images(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id=1, subset="a", media=Image(data=np.ones([4, 5, 3]))),
                        DatasetItem(id=2, subset="a", media=Image(data=np.ones([4, 5, 3]))),
                        DatasetItem(id=3, subset="b", media=Image(data=np.ones([2, 6, 3]))),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", save_media=True, tasks=task),
                    test_dir,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_with_voc_labelmap(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=self._label("cat"), id=1),
                        Bbox(1, 2, 3, 4, label=self._label("non_voc_label"), id=2),
                    ],
                )

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add(VOC.VocLabel.cat.name)
                label_cat.add("non_voc_label")
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        # drop non voc label
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=self._label("cat"),
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )

            def categories(self):
                return VOC.make_voc_categories()

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(VocExporter.convert, label_map="voc"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_with_source_labelmap_undefined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=0, id=1),
                        Bbox(1, 2, 3, 4, label=1, id=2),
                    ],
                )

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add("Label_1")
                label_cat.add("label_2")
                return {
                    AnnotationType.label: label_cat,
                }

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=self._label("Label_1"),
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=self._label("label_2"),
                            id=2,
                            group=2,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["background"] = [None, [], []]
                label_map["Label_1"] = [None, [], []]
                label_map["label_2"] = [None, [], []]
                return VOC.make_voc_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(VocExporter.convert, label_map="source"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_with_source_labelmap_defined(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=0, id=1),
                        Bbox(1, 2, 3, 4, label=2, id=2),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["label_1"] = [(1, 2, 3), [], []]
                label_map["background"] = [(0, 0, 0), [], []]  # can be not 0
                label_map["label_2"] = [(3, 2, 1), [], []]
                return VOC.make_voc_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=self._label("label_1"),
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=self._label("label_2"),
                            id=2,
                            group=2,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )

            def categories(self):
                label_map = OrderedDict()
                label_map["label_1"] = [(1, 2, 3), [], []]
                label_map["background"] = [(0, 0, 0), [], []]
                label_map["label_2"] = [(3, 2, 1), [], []]
                return VOC.make_voc_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(VocExporter.convert, label_map="source"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_with_save_dataset_meta_file(self):
        label_map = OrderedDict(
            [
                ("background", [(0, 0, 0), [], []]),
                ("label_1", [(1, 2, 3), ["part1", "part2"], ["act1", "act2"]]),
                ("label_2", [(3, 2, 1), ["part3"], []]),
            ]
        )

        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=1, id=1),
                    ],
                )

            def categories(self):
                return VOC.make_voc_categories(label_map)

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            2,
                            3,
                            4,
                            5,
                            label=self._label("label_1"),
                            id=1,
                            group=1,
                            attributes={
                                "act1": False,
                                "act2": False,
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                    ],
                )

            def categories(self):
                return VOC.make_voc_categories(label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(VocExporter.convert, label_map=label_map, save_dataset_meta=True),
                test_dir,
                target_dataset=DstExtractor(),
            )
            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset_with_fixed_labelmap(self):
        class SrcExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(2, 3, 4, 5, label=self._label("foreign_label"), id=1),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=self._label("label"),
                            id=2,
                            group=2,
                            attributes={"act1": True},
                        ),
                        Bbox(2, 3, 4, 5, label=self._label("label_part1"), group=2),
                        Bbox(2, 3, 4, 6, label=self._label("label_part2"), group=2),
                    ],
                )

            def categories(self):
                label_cat = LabelCategories()
                label_cat.add("foreign_label")
                label_cat.add("label", attributes=["act1", "act2"])
                label_cat.add("label_part1")
                label_cat.add("label_part2")
                return {
                    AnnotationType.label: label_cat,
                }

        label_map = OrderedDict(
            [("label", [None, ["label_part1", "label_part2"], ["act1", "act2"]])]
        )

        dst_label_map = OrderedDict(
            [
                ("background", [None, [], []]),
                ("label", [None, ["label_part1", "label_part2"], ["act1", "act2"]]),
            ]
        )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                yield DatasetItem(
                    id=1,
                    annotations=[
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=self._label("label"),
                            id=1,
                            group=1,
                            attributes={
                                "act1": True,
                                "act2": False,
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Bbox(2, 3, 4, 5, label=self._label("label_part1"), group=1),
                        Bbox(2, 3, 4, 6, label=self._label("label_part2"), group=1),
                    ],
                )

            def categories(self):
                return VOC.make_voc_categories(dst_label_map)

        with TestDir() as test_dir:
            self._test_save_and_load(
                SrcExtractor(),
                partial(VocExporter.convert, label_map=label_map),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_background_masks_dont_introduce_instances_but_cover_others(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    media=Image(data=np.zeros((4, 1, 1))),
                    annotations=[
                        Mask([1, 1, 1, 1], label=1, attributes={"z_order": 1}),
                        Mask([0, 0, 1, 1], label=2, attributes={"z_order": 2}),
                        Mask([0, 0, 1, 1], label=0, attributes={"z_order": 3}),
                    ],
                )
            ],
            categories=["background", "a", "b"],
        )

        with TestDir() as test_dir:
            VocExporter.convert(dataset, test_dir, apply_colormap=False)

            cls_mask = load_mask(osp.join(test_dir, "SegmentationClass", "1.png"))
            inst_mask = load_mask(osp.join(test_dir, "SegmentationObject", "1.png"))
            self.assertTrue(np.array_equal([0, 1], np.unique(cls_mask)))
            self.assertTrue(np.array_equal([0, 1], np.unique(inst_mask)))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id=1, media=Image(path="1.jpg", size=(10, 15))),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", tasks=task),
                    test_dir,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="q/1", media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3)))
                        ),
                        DatasetItem(
                            id="a/b/c/2", media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3)))
                        ),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", tasks=task, save_media=True),
                    test_dir,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(id="1", media=Image(data=np.ones((4, 2, 3)))),
                        DatasetItem(id="subdir1/1", media=Image(data=np.ones((2, 6, 3)))),
                        DatasetItem(id="subdir2/1", media=Image(data=np.ones((5, 4, 3)))),
                    ]
                )

        for task in [None] + list(VOC.VocTask):
            with self.subTest(subformat=task), TestDir() as test_dir:
                self._test_save_and_load(
                    TestExtractor(),
                    partial(VocExporter.convert, label_map="voc", save_media=True, tasks=task),
                    test_dir,
                    require_media=True,
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_attributes(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    attributes={"occluded": True, "x": 1, "y": "2"},
                                ),
                            ],
                        ),
                    ]
                )

        class DstExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="a",
                            annotations=[
                                Bbox(
                                    2,
                                    3,
                                    4,
                                    5,
                                    label=2,
                                    id=1,
                                    group=1,
                                    attributes={
                                        "truncated": False,
                                        "difficult": False,
                                        "occluded": True,
                                        "x": "1",
                                        "y": "2",  # can only read strings
                                    },
                                ),
                            ],
                        ),
                    ]
                )

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(),
                partial(VocExporter.convert, label_map="voc"),
                test_dir,
                target_dataset=DstExtractor(),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image(data=np.ones((1, 2, 3))),
                    annotations=[
                        # Bbox(0, 0, 0, 0, label=1) # won't find removed anns
                    ],
                ),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image(data=np.ones((3, 2, 3))),
                    annotations=[
                        Bbox(
                            0,
                            0,
                            0,
                            0,
                            label=4,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        )
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["background", "a", "b", "c", "d"]
                ),
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(5)),
            },
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image(data=np.ones((1, 2, 3))),
                    annotations=[Bbox(0, 0, 0, 0, label=1)],
                ),
                DatasetItem(2, subset="b", annotations=[Bbox(0, 0, 0, 0, label=2)]),
                DatasetItem(
                    3,
                    subset="c",
                    media=Image(data=np.ones((2, 2, 3))),
                    annotations=[Bbox(0, 0, 0, 0, label=3), Mask(np.ones((2, 2)), label=1)],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        with TestDir() as path:
            dataset.export(path, "voc", save_media=True)
            os.unlink(osp.join(path, "Annotations", "1.xml"))
            os.unlink(osp.join(path, "Annotations", "2.xml"))
            os.unlink(osp.join(path, "Annotations", "3.xml"))

            dataset.put(
                DatasetItem(
                    2,
                    subset="b",
                    media=Image(data=np.ones((3, 2, 3))),
                    annotations=[Bbox(0, 0, 0, 0, label=3)],
                )
            )
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual(
                {"2.xml"},  # '1.xml' won't be touched
                set(os.listdir(osp.join(path, "Annotations"))),
            )
            self.assertEqual({"1.jpg", "2.jpg"}, set(os.listdir(osp.join(path, "JPEGImages"))))
            self.assertEqual(
                {"a.txt", "b.txt"}, set(os.listdir(osp.join(path, "ImageSets", "Main")))
            )
            compare_datasets(self, expected, Dataset.import_from(path, "voc"), require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    3,
                    subset="test",
                    media=Image(data=np.ones((2, 3, 3))),
                    annotations=[
                        Bbox(
                            0,
                            1,
                            0,
                            0,
                            label=4,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        )
                    ],
                ),
                DatasetItem(
                    4,
                    subset="train",
                    media=Image(data=np.ones((2, 4, 3))),
                    annotations=[
                        Bbox(
                            1,
                            0,
                            0,
                            0,
                            label=4,
                            id=1,
                            group=1,
                            attributes={
                                "truncated": False,
                                "difficult": False,
                                "occluded": False,
                            },
                        ),
                        Mask(np.ones((2, 2)), label=2, group=1),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    ["background", "a", "b", "c", "d"]
                ),
                AnnotationType.mask: MaskCategories(colormap=VOC.generate_colormap(5)),
            },
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    subset="a",
                    media=Image(data=np.ones((2, 1, 3))),
                    annotations=[Bbox(0, 0, 0, 1, label=1)],
                ),
                DatasetItem(
                    2,
                    subset="b",
                    media=Image(data=np.ones((2, 2, 3))),
                    annotations=[
                        Bbox(0, 0, 1, 0, label=2),
                        Mask(np.ones((2, 2)), label=1),
                    ],
                ),
                DatasetItem(
                    3,
                    subset="b",
                    media=Image(data=np.ones((2, 3, 3))),
                    annotations=[Bbox(0, 1, 0, 0, label=3)],
                ),
                DatasetItem(
                    4,
                    subset="c",
                    media=Image(data=np.ones((2, 4, 3))),
                    annotations=[Bbox(1, 0, 0, 0, label=3), Mask(np.ones((2, 2)), label=1)],
                ),
            ],
            categories=["a", "b", "c", "d"],
        )

        with TestDir() as path:
            dataset.export(path, "voc", save_media=True)

            dataset.filter("/item[id >= 3]")
            dataset.transform("random_split", splits=(("train", 0.5), ("test", 0.5)), seed=42)
            dataset.save(save_media=True)

            self.assertEqual({"3.xml", "4.xml"}, set(os.listdir(osp.join(path, "Annotations"))))
            self.assertEqual({"3.jpg", "4.jpg"}, set(os.listdir(osp.join(path, "JPEGImages"))))
            self.assertEqual({"4.png"}, set(os.listdir(osp.join(path, "SegmentationClass"))))
            self.assertEqual({"4.png"}, set(os.listdir(osp.join(path, "SegmentationObject"))))
            self.assertEqual(
                {"train.txt", "test.txt"}, set(os.listdir(osp.join(path, "ImageSets", "Main")))
            )
            self.assertEqual(
                {"train.txt"}, set(os.listdir(osp.join(path, "ImageSets", "Segmentation")))
            )
            compare_datasets(self, expected, Dataset.import_from(path, "voc"), require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_no_data_images(self):
        class TestExtractor(TestExtractorBase):
            def __iter__(self):
                return iter(
                    [
                        DatasetItem(
                            id="frame1",
                            subset="test",
                            media=Image(path="frame1.jpg"),
                            annotations=[
                                Bbox(
                                    1.0,
                                    2.0,
                                    3.0,
                                    4.0,
                                    attributes={
                                        "difficult": False,
                                        "truncated": False,
                                        "occluded": False,
                                    },
                                    id=1,
                                    label=0,
                                    group=1,
                                )
                            ],
                        )
                    ]
                )

            def categories(self):
                return VOC.make_voc_categories()

        with TestDir() as test_dir:
            self._test_save_and_load(
                TestExtractor(), partial(VocExporter.convert, label_map="voc"), test_dir
            )
