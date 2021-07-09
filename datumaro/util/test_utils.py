# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from glob import glob
import inspect
import os
import os.path as osp
import tempfile

from datumaro.components.dataset import Dataset
from datumaro.components.extractor import AnnotationType
from datumaro.util import filter_dict, find
from datumaro.util.os_util import rmfile, rmtree


class Dimensions(Enum):
    dim_2d = auto()
    dim_3d = auto()

def current_function_name(depth=1):
    return inspect.getouterframes(inspect.currentframe())[depth].function

class FileRemover:
    def __init__(self, path, is_dir=False):
        self.path = path
        self.is_dir = is_dir

    def __enter__(self):
        return self.path

    # pylint: disable=redefined-builtin
    def __exit__(self, type=None, value=None, traceback=None):
        if self.is_dir:
            rmtree(self.path)
        else:
            rmfile(self.path)
    # pylint: enable=redefined-builtin

class TestDir(FileRemover):
    """
    Creates a temporary directory for a test. Uses the name of
    the test function to name the directory.

    Usage:

    with TestDir() as test_dir:
        ...
    """

    def __init__(self, path=None):
        if path is None:
            path = osp.abspath('temp_%s-' % current_function_name(2))
            path = tempfile.mkdtemp(dir=os.getcwd(), prefix=path)
        else:
            os.makedirs(path, exist_ok=False)

        super().__init__(path, is_dir=True)

def compare_categories(test, expected, actual):
    test.assertEqual(
        sorted(expected, key=lambda t: t.value),
        sorted(actual, key=lambda t: t.value)
    )

    if AnnotationType.label in expected:
        test.assertEqual(
            expected[AnnotationType.label].items,
            actual[AnnotationType.label].items,
        )
    if AnnotationType.mask in expected:
        test.assertEqual(
            expected[AnnotationType.mask].colormap,
            actual[AnnotationType.mask].colormap,
        )
    if AnnotationType.points in expected:
        test.assertEqual(
            expected[AnnotationType.points].items,
            actual[AnnotationType.points].items,
        )

def _compare_annotations(expected, actual, ignored_attrs=None):
    if not ignored_attrs:
        return expected == actual

    a_attr = expected.attributes
    b_attr = actual.attributes

    expected.attributes = filter_dict(a_attr, exclude_keys=ignored_attrs)
    actual.attributes = filter_dict(b_attr, exclude_keys=ignored_attrs)
    r = expected == actual

    expected.attributes = a_attr
    actual.attributes = b_attr
    return r

def compare_datasets(test, expected, actual, ignored_attrs=None,
        require_images=False):
    compare_categories(test, expected.categories(), actual.categories())

    test.assertEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))
    for item_a in expected:
        item_b = find(actual, lambda x: x.id == item_a.id and \
            x.subset == item_a.subset)
        test.assertFalse(item_b is None, item_a.id)

        if ignored_attrs:
            test.assertEqual(item_a.attributes,
                filter_dict(item_b.attributes, exclude_keys=ignored_attrs),
                item_a.id)
        else:
            test.assertEqual(item_a.attributes, item_b.attributes, item_a.id)

        if (require_images and item_a.has_image and item_a.image.has_data) or \
                item_a.has_image and item_a.image.has_data and \
                item_b.has_image and item_b.image.has_data:
            test.assertEqual(item_a.image, item_b.image, item_a.id)
        test.assertEqual(len(item_a.annotations), len(item_b.annotations),
            item_a.id)
        for ann_a in item_a.annotations:
            # We might find few corresponding items, so check them all
            ann_b_matches = [x for x in item_b.annotations
                if x.type == ann_a.type]
            test.assertFalse(len(ann_b_matches) == 0, 'ann id: %s' % ann_a.id)

            ann_b = find(ann_b_matches, lambda x:
                _compare_annotations(x, ann_a, ignored_attrs=ignored_attrs))
            if ann_b is None:
                test.fail('ann %s, candidates %s' % (ann_a, ann_b_matches))
            item_b.annotations.remove(ann_b) # avoid repeats

def compare_datasets_strict(test, expected, actual):
    # Compares datasets for strong equality

    test.assertEqual(expected.categories(), actual.categories())

    test.assertListEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))

    for subset_name in expected.subsets():
        e_subset = expected.get_subset(subset_name)
        a_subset = actual.get_subset(subset_name)
        test.assertEqual(len(e_subset), len(a_subset))
        for idx, (item_a, item_b) in enumerate(zip(e_subset, a_subset)):
            test.assertEqual(item_a, item_b,
                '%s:\n%s\nvs.\n%s\n' % \
                (idx, item_a, item_b))

def compare_datasets_3d(test, expected, actual, ignored_attrs=None,
        require_point_cloud=False):
    compare_categories(test, expected.categories(), actual.categories())

    if actual.subsets():
        test.assertEqual(sorted(expected.subsets()), sorted(actual.subsets()))

    test.assertEqual(len(expected), len(actual))
    for item_a in expected:
        item_b = find(actual, lambda x: x.id == item_a.id)
        test.assertFalse(item_b is None, item_a.id)

        if ignored_attrs:
            test.assertEqual(item_a.attributes,
                filter_dict(item_b.attributes, exclude_keys=ignored_attrs),
                item_a.id)
        else:
            test.assertEqual(item_a.attributes, item_b.attributes, item_a.id)

        if (require_point_cloud and item_a.has_point_cloud) or \
                (item_a.has_point_cloud and item_b.has_point_cloud):
            test.assertEqual(item_a.point_cloud, item_b.point_cloud, item_a.id)
            test.assertEqual(
                set(img.path for img in item_a.related_images),
                set(img.path for img in item_b.related_images),
                item_a.id)
        test.assertEqual(len(item_a.annotations), len(item_b.annotations))
        for ann_a in item_a.annotations:
            # We might find few corresponding items, so check them all
            ann_b_matches = [x for x in item_b.annotations
                if x.type == ann_a.type]
            test.assertFalse(len(ann_b_matches) == 0, 'ann id: %s' % ann_a.id)

            ann_b = find(ann_b_matches, lambda x:
                _compare_annotations(x, ann_a, ignored_attrs=ignored_attrs))
            if ann_b is None:
                test.fail('ann %s, candidates %s' % (ann_a, ann_b_matches))
            item_b.annotations.remove(ann_b) # avoid repeats


def test_save_and_load(test, source_dataset, converter, test_dir, importer,
        target_dataset=None, importer_args=None, compare=None, **kwargs):
    converter(source_dataset, test_dir)

    if importer_args is None:
        importer_args = {}
    parsed_dataset = Dataset.import_from(test_dir, importer, **importer_args)

    if target_dataset is None:
        target_dataset = source_dataset

    if not compare and kwargs.get("dimension") is Dimensions.dim_3d:
        compare = compare_datasets_3d
        del kwargs["dimension"]
    elif not compare:
        compare = compare_datasets
    compare(test, expected=target_dataset, actual=parsed_dataset, **kwargs)

def compare_dirs(test, expected, actual, skip_empty_dirs=True):
    for a_path in glob(osp.join(expected, '**', '*'), recursive=True):
        rel_path = osp.relpath(a_path, expected)
        b_path = osp.join(actual, rel_path)
        if osp.isdir(a_path):
            if not (skip_empty_dirs and not os.listdir(a_path)):
                test.assertTrue(osp.isdir(b_path), rel_path)
            continue

        test.assertTrue(osp.isfile(b_path), rel_path)
        with open(a_path, 'rb') as a_file, \
                open(b_path, 'rb') as b_file:
            test.assertEqual(a_file.read(), b_file.read(), rel_path)


def run_datum(test, *args, expected_code=0):
    from datumaro.cli.__main__ import main
    test.assertEqual(expected_code, main(args), str(args))