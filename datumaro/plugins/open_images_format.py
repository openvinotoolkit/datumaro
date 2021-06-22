# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import csv
import fnmatch
import glob
import json
import os
import os.path as osp
import re

from attr import attrs

from datumaro.components.errors import DatasetError, RepeatedItemError, UndefinedLabel
from datumaro.components.extractor import (
    AnnotationType, DatasetItem, Importer, Label, LabelCategories, Extractor,
)
from datumaro.components.validator import Severity
from datumaro.util.image import find_images

# A regex to check whether a subset name can be used as a "normal" path
# component.
# Accepting a subset name that doesn't match this regex could lead
# to accessing data outside of the expected directory, so it's best
# to reject them.
_RE_INVALID_SUBSET = re.compile(r'''
    # empty
    | \.\.? # special path component
    | .*[/\\\0].* # contains special characters
''', re.VERBOSE)

@attrs(auto_attribs=True)
class UnsupportedSubsetNameError(DatasetError):
    subset: str

    def __str__(self):
        return "Item %s has an unsupported subset name %r." % (self.item_id, self.subset)

class OpenImagesPath:
    ANNOTATIONS_DIR = 'annotations'
    FULL_IMAGE_DESCRIPTION_NAME = 'image_ids_and_rotation.csv'
    SUBSET_IMAGE_DESCRIPTION_PATTERNS = (
        '*-images-with-rotation.csv',
        '*-images-with-labels-with-rotation.csv',
    )

class OpenImagesExtractor(Extractor):
    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Can't read dataset directory '%s'" % path)

        super().__init__()

        self._dataset_dir = path

        self._annotation_files = os.listdir(
            osp.join(path, OpenImagesPath.ANNOTATIONS_DIR))

        self._categories = {}
        self._items = []

        self._load_categories()
        self._load_items()

    def __iter__(self):
        return iter(self._items)

    def categories(self):
        return self._categories

    @contextlib.contextmanager
    def _open_csv_annotation(self, file_name):
        absolute_path = osp.join(self._dataset_dir, OpenImagesPath.ANNOTATIONS_DIR, file_name)

        with open(absolute_path, 'r', newline='') as f:
            yield csv.DictReader(f)

    def _glob_annotations(self, pattern):
        for annotation_file in self._annotation_files:
            if fnmatch.fnmatch(annotation_file, pattern):
                yield annotation_file

    def _load_categories(self):
        label_categories = LabelCategories()

        with self._open_csv_annotation('oidv6-class-descriptions.csv') as class_description_reader:
            for class_description in class_description_reader:
                label_categories.add(class_description['LabelName'])

        self._categories[AnnotationType.label] = label_categories

        self._load_label_category_parents()

    def _load_label_category_parents(self):
        label_categories = self._categories[AnnotationType.label]

        hierarchy_path = osp.join(
            self._dataset_dir, OpenImagesPath.ANNOTATIONS_DIR, 'bbox_labels_600_hierarchy.json')

        try:
            with open(hierarchy_path, 'rb') as hierarchy_file:
                root_node = json.load(hierarchy_file)
        except FileNotFoundError:
            return

        def set_parents_from_node(node, category):
            for child_node in node.get('Subcategory', []):
                _, child_category = label_categories.find(child_node['LabelName'])

                if category is not None and child_category is not None:
                    child_category.parent = category.name

                set_parents_from_node(child_node, child_category)

        _, root_category = label_categories.find(root_node['LabelName'])
        set_parents_from_node(root_node, root_category)

    def _load_items(self):
        image_paths_by_id = {
            osp.splitext(osp.basename(path))[0]: path
            for path in find_images(
                osp.join(self._dataset_dir, 'images'),
                recursive=True, max_depth=1)
        }

        items_by_id = {}

        def load_from(annotation_name):
            with self._open_csv_annotation(annotation_name) as image_reader:
                for image_description in image_reader:
                    image_id = image_description['ImageID']
                    if image_id in items_by_id:
                        raise RepeatedItemError(item_id=image_id)

                    subset = image_description['Subset']

                    if _RE_INVALID_SUBSET.fullmatch(subset):
                        raise UnsupportedSubsetNameError(item_id=image_id, subset=subset)

                    items_by_id[image_id] = DatasetItem(
                        id=image_id,
                        image=image_paths_by_id.get(image_id),
                        subset=subset,
                    )

        # It's preferable to load the combined image description file,
        # because it contains descriptions for training images without human-annotated labels
        # (the file specific to the training set doesn't).
        # However, if it's missing, we'll try loading subset-specific files instead, so that
        # this extractor can be used on individual subsets of the dataset.
        try:
            load_from(OpenImagesPath.FULL_IMAGE_DESCRIPTION_NAME)
        except FileNotFoundError:
            for pattern in OpenImagesPath.SUBSET_IMAGE_DESCRIPTION_PATTERNS:
                for path in self._glob_annotations(pattern):
                    load_from(path)

        self._items.extend(items_by_id.values())

        self._load_labels(items_by_id)

    def _load_labels(self, items_by_id):
        label_categories = self._categories[AnnotationType.label]

        # TODO: implement reading of machine-annotated labels

        for label_path in self._glob_annotations('*-human-imagelabels.csv'):
            with self._open_csv_annotation(label_path) as label_reader:
                for label_description in label_reader:
                    image_id = label_description['ImageID']
                    item = items_by_id[image_id]

                    confidence = float(label_description['Confidence'])

                    if 0.5 < confidence:
                        label_name = label_description['LabelName']
                        label_index, _ = label_categories.find(label_name)
                        if label_index is None:
                            raise UndefinedLabel(
                                item_id=item.id, subset=item.subset,
                                label_name=label_name, severity=Severity.error)
                        item.annotations.append(Label(label_index))


class OpenImagesImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        for pattern in [
            OpenImagesPath.FULL_IMAGE_DESCRIPTION_NAME,
            *OpenImagesPath.SUBSET_IMAGE_DESCRIPTION_PATTERNS,
        ]:
            if glob.glob(osp.join(glob.escape(path), OpenImagesPath.ANNOTATIONS_DIR, pattern)):
                return [{'url': path, 'format': 'open_images'}]

        return []
