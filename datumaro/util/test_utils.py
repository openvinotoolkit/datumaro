# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import inspect
import os
import os.path as osp
import tempfile
import unittest
import unittest.mock
import warnings
from enum import Enum, auto
from glob import glob
from typing import Collection, Optional, Union

from typing_extensions import Literal

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset, IDataset
from datumaro.components.media import Image, MultiframeImage, PointCloud
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

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        if self.is_dir:
            try:
                rmtree(self.path)
            except unittest.SkipTest:
                # Suppress skip test errors from git.util.rmtree
                if not exc_type:
                    raise
        else:
            rmfile(self.path)


class TestDir(FileRemover):
    """
    Creates a temporary directory for a test. Uses the name of
    the test function to name the directory.

    Usage:

    .. code-block::

        with TestDir() as test_dir:
            ...
    """

    def __init__(self, path: Optional[str] = None, frame_id: int = 2):
        if not path:
            prefix = f"temp_{current_function_name(frame_id)}-"
        else:
            prefix = None
        self._prefix = prefix

        super().__init__(path, is_dir=True)

    def __enter__(self) -> str:
        """
        Creates a test directory.

        Returns: path to the directory
        """

        path = self.path

        if path is None:
            path = tempfile.mkdtemp(dir=os.getcwd(), prefix=self._prefix)
            self.path = path
        else:
            os.makedirs(path, exist_ok=False)

        return path


def compare_categories(test, expected, actual):
    test.assertEqual(sorted(expected, key=lambda t: t.value), sorted(actual, key=lambda t: t.value))

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


IGNORE_ALL = "*"


def _compare_annotations(expected, actual, ignored_attrs=None):
    if not ignored_attrs:
        return expected == actual

    a_attr = expected.attributes
    b_attr = actual.attributes

    if ignored_attrs != IGNORE_ALL:
        expected.attributes = filter_dict(a_attr, exclude_keys=ignored_attrs)
        actual.attributes = filter_dict(b_attr, exclude_keys=ignored_attrs)
    else:
        expected.attributes = {}
        actual.attributes = {}

    r = expected == actual

    expected.attributes = a_attr
    actual.attributes = b_attr

    return r


def compare_datasets(
    test,
    expected: IDataset,
    actual: IDataset,
    ignored_attrs: Union[None, Literal["*"], Collection[str]] = None,
    require_media: bool = False,
    require_images: bool = False,
):
    compare_categories(test, expected.categories(), actual.categories())

    test.assertTrue(issubclass(actual.media_type(), expected.media_type()))

    test.assertEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))

    for item_a in expected:
        item_b = find(actual, lambda x: x.id == item_a.id and x.subset == item_a.subset)
        test.assertFalse(item_b is None, item_a.id)

        if ignored_attrs and ignored_attrs != IGNORE_ALL:
            test.assertEqual(
                item_a.attributes,
                filter_dict(item_b.attributes, exclude_keys=ignored_attrs),
                item_a.id,
            )
        elif not ignored_attrs:
            test.assertEqual(item_a.attributes, item_b.attributes, item_a.id)

        if require_images:
            warnings.warn(
                "'require_images' is deprecated and will be "
                "removed in future. Use 'require_media' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        require_media = require_media or require_images

        if require_media and item_a.media and item_b.media:
            if isinstance(item_a.media, Image):
                test.assertEqual(item_a.media, item_b.media, item_a.id)
            elif isinstance(item_a.media, PointCloud):
                test.assertEqual(item_a.media.extra_images, item_b.media.extra_images, item_a.id)
            elif isinstance(item_a.media, MultiframeImage):
                test.assertEqual(item_a.media.data, item_b.media.data, item_a.id)
        test.assertEqual(len(item_a.annotations), len(item_b.annotations), item_a.id)
        for ann_a in item_a.annotations:
            # We might find few corresponding items, so check them all
            ann_b_matches = [x for x in item_b.annotations if x.type == ann_a.type]
            test.assertFalse(len(ann_b_matches) == 0, "ann id: %s" % ann_a.id)

            ann_b = find(
                ann_b_matches, lambda x: _compare_annotations(x, ann_a, ignored_attrs=ignored_attrs)
            )
            if ann_b is None:
                test.fail("ann %s, candidates %s" % (ann_a, ann_b_matches))
            item_b.annotations.remove(ann_b)  # avoid repeats


def compare_datasets_strict(test, expected: IDataset, actual: IDataset):
    # Compares datasets for strong equality

    test.assertEqual(expected.media_type(), actual.media_type())
    test.assertEqual(expected.categories(), actual.categories())

    test.assertListEqual(sorted(expected.subsets()), sorted(actual.subsets()))
    test.assertEqual(len(expected), len(actual))

    for subset_name in expected.subsets():
        e_subset = expected.get_subset(subset_name)
        a_subset = actual.get_subset(subset_name)
        test.assertEqual(len(e_subset), len(a_subset))
        for idx, (item_a, item_b) in enumerate(zip(e_subset, a_subset)):
            test.assertEqual(item_a, item_b, "%s:\n%s\nvs.\n%s\n" % (idx, item_a, item_b))


def compare_datasets_3d(
    test,
    expected: IDataset,
    actual: IDataset,
    ignored_attrs: Union[None, Literal["*"], Collection[str]] = None,
    require_point_cloud: bool = False,
):
    compare_categories(test, expected.categories(), actual.categories())

    if actual.subsets():
        test.assertEqual(sorted(expected.subsets()), sorted(actual.subsets()))

    test.assertEqual(len(expected), len(actual))
    for item_a in expected:
        item_b = find(actual, lambda x: x.id == item_a.id)
        test.assertFalse(item_b is None, item_a.id)

        if ignored_attrs and ignored_attrs != IGNORE_ALL:
            test.assertEqual(
                item_a.attributes,
                filter_dict(item_b.attributes, exclude_keys=ignored_attrs),
                item_a.id,
            )
        elif not ignored_attrs:
            test.assertEqual(item_a.attributes, item_b.attributes, item_a.id)

        if (require_point_cloud and item_a.media) or (item_a.media and item_b.media):
            test.assertEqual(item_a.media.path, item_b.media.path, item_a.id)
            test.assertEqual(
                set(img.path for img in item_a.media.extra_images),
                set(img.path for img in item_b.media.extra_images),
                item_a.id,
            )
        test.assertEqual(len(item_a.annotations), len(item_b.annotations))
        for ann_a in item_a.annotations:
            # We might find few corresponding items, so check them all
            ann_b_matches = [x for x in item_b.annotations if x.type == ann_a.type]
            test.assertFalse(len(ann_b_matches) == 0, "ann id: %s" % ann_a.id)

            ann_b = find(
                ann_b_matches, lambda x: _compare_annotations(x, ann_a, ignored_attrs=ignored_attrs)
            )
            if ann_b is None:
                test.fail("ann %s, candidates %s" % (ann_a, ann_b_matches))
            item_b.annotations.remove(ann_b)  # avoid repeats


def check_save_and_load(
    test,
    source_dataset,
    converter,
    test_dir,
    importer,
    target_dataset=None,
    importer_args=None,
    compare=None,
    **cmp_kwargs,
):
    converter(source_dataset, test_dir)

    if importer_args is None:
        importer_args = {}
    parsed_dataset = Dataset.import_from(test_dir, importer, **importer_args)

    if target_dataset is None:
        target_dataset = source_dataset

    if not compare and cmp_kwargs.get("dimension") is Dimensions.dim_3d:
        compare = compare_datasets_3d
        del cmp_kwargs["dimension"]
    elif not compare:
        compare = compare_datasets
    compare(test, expected=target_dataset, actual=parsed_dataset, **cmp_kwargs)


def compare_dirs(test, expected: str, actual: str):
    """
    Compares file and directory structures in the given directories.
    Empty directories are skipped.
    """
    skip_empty_dirs = True

    for a_path in glob(osp.join(expected, "**", "*"), recursive=True):
        rel_path = osp.relpath(a_path, expected)
        b_path = osp.join(actual, rel_path)
        if osp.isdir(a_path):
            if not (skip_empty_dirs and not os.listdir(a_path)):
                test.assertTrue(osp.isdir(b_path), rel_path)
            continue

        test.assertTrue(osp.isfile(b_path), rel_path)
        with open(a_path, "rb") as a_file, open(b_path, "rb") as b_file:
            test.assertEqual(a_file.read(), b_file.read(), rel_path)


def run_datum(test, *args, expected_code=0):
    from datumaro.cli.__main__ import main

    test.assertEqual(expected_code, main(args), str(args))


@contextlib.contextmanager
def mock_tfds_data(example=None, subsets=("train",)):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    NUM_EXAMPLES = 1

    if example:

        def as_dataset(self, *args, **kwargs):
            return tf.data.Dataset.from_tensors(example)

    else:
        as_dataset = None

    with tfds.testing.mock_data(num_examples=NUM_EXAMPLES, as_dataset_fn=as_dataset):
        # The mock version of DatasetBuilder.__init__ installed by mock_data
        # doesn't initialize split info, which TfdsExtractor needs to function.
        # So we mock it again to fix that. See also TFDS feature request at
        # <https://github.com/tensorflow/datasets/issues/3631>.
        original_init = tfds.core.DatasetBuilder.__init__

        def new_init(self, **kwargs):
            original_init(self, **kwargs)
            self.info.set_splits(
                tfds.core.SplitDict(
                    [
                        tfds.core.SplitInfo(
                            name=subset_name, shard_lengths=[NUM_EXAMPLES], num_bytes=1234
                        )
                        for subset_name in subsets
                    ],
                    dataset_name=self.name,
                ),
            )

        with unittest.mock.patch("tensorflow_datasets.core.DatasetBuilder.__init__", new_init):
            yield
