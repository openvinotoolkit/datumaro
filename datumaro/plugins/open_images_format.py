# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import contextlib
import csv
import fnmatch
import glob
import itertools
import json
import os
import os.path as osp
import re

from attr import attrs

from datumaro.components.converter import Converter
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
    V5_CLASS_DESCRIPTION_NAME = 'class-descriptions.csv'
    HIERARCHY_NAME = 'bbox_labels_600_hierarchy.json'

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

        with open(absolute_path, 'r', encoding='utf-8', newline='') as f:
            yield csv.DictReader(f)

    def _glob_annotations(self, pattern):
        for annotation_file in self._annotation_files:
            if fnmatch.fnmatch(annotation_file, pattern):
                yield annotation_file

    def _load_categories(self):
        label_categories = LabelCategories()

        # In OID v6, the class description file is prefixed with `oidv6-`, whereas
        # in the previous versions, it isn't. We try to find it regardless.
        # We use a wildcard so that if, say, OID v7 is released in the future with
        # a similar layout as v6, it's automatically supported.
        # If the file doesn't exist with either name, we'll fail trying to open
        # `class-descriptions.csv`.

        annotation_name = [
            *self._glob_annotations('oidv*-class-descriptions.csv'),
            OpenImagesPath.V5_CLASS_DESCRIPTION_NAME,
        ][0]

        with self._open_csv_annotation(annotation_name) as class_description_reader:
            # Prior to OID v6, this file didn't contain a header row.
            if annotation_name == OpenImagesPath.V5_CLASS_DESCRIPTION_NAME:
                class_description_reader.fieldnames = ('LabelName', 'DisplayName')

            for class_description in class_description_reader:
                label_name = class_description['LabelName']
                label_categories.add(label_name)

        self._categories[AnnotationType.label] = label_categories

        self._load_label_category_parents()

    def _load_label_category_parents(self):
        label_categories = self._categories[AnnotationType.label]

        hierarchy_path = osp.join(
            self._dataset_dir, OpenImagesPath.ANNOTATIONS_DIR, OpenImagesPath.HIERARCHY_NAME)

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

                    label_name = label_description['LabelName']
                    label_index, _ = label_categories.find(label_name)
                    if label_index is None:
                        raise UndefinedLabel(
                            item_id=item.id, subset=item.subset,
                            label_name=label_name, severity=Severity.error)
                    item.annotations.append(Label(
                        label=label_index, attributes={'score': confidence}))


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

class OpenImagesConverter(Converter):
    DEFAULT_IMAGE_EXT = '.jpg'

    @contextlib.contextmanager
    def _open_csv_annotation(self, file_name, field_names):
        absolute_path = osp.join(self._save_dir, OpenImagesPath.ANNOTATIONS_DIR, file_name)

        with open(absolute_path, 'w', encoding='utf-8', newline='') as f:
            yield csv.DictWriter(f, field_names)

    def apply(self):
        annotations_dir = osp.join(self._save_dir, OpenImagesPath.ANNOTATIONS_DIR)

        os.makedirs(annotations_dir, exist_ok=True)

        self._save_categories()
        self._save_label_category_parents()
        self._save_subsets()

    def _save_categories(self):
        with self._open_csv_annotation(
            OpenImagesPath.V5_CLASS_DESCRIPTION_NAME, ['LabelName', 'DisplayName'],
        ) as class_description_writer:
            # no .writeheader() here, since we're saving it in the V5 format

            for category in self._extractor.categories()[AnnotationType.label]:
                class_description_writer.writerow({
                    'LabelName': category.name,
                    'DisplayName': category.name,
                })

    def _save_label_category_parents(self):
        all_label_names = set()
        hierarchy_nodes = {}
        orphan_nodes = []

        def get_node(name):
            return hierarchy_nodes.setdefault(name, {'LabelName': name})

        for category in self._extractor.categories()[AnnotationType.label]:
            all_label_names.add(category.name)

            child_node = get_node(category.name)

            if category.parent:
                parent_node = get_node(category.parent)
                parent_node.setdefault('Subcategory', []).append(child_node)
            else:
                orphan_nodes.append(child_node)

        # The hierarchy has to be rooted in a single node. However, there's
        # no guarantee that there exists only one orphan (label without a parent).
        # Therefore, we create a fake root node and make it the parent of every
        # orphan label.
        # This is not a violation of the format, because the original OID does
        # the same thing.
        root_node = {
            # Create an OID-like label name that isn't already used by a real label
            'LabelName': next(root_name
                for i in itertools.count()
                for root_name in [f'/m/{i}']
                if root_name not in all_label_names
            ),
            # If an orphan has no children, then it makes no semantic difference
            # whether it's listed in the hierarchy file or not. So strip such nodes
            # to avoid recording meaningless data.
            'Subcategory': [node for node in orphan_nodes if 'Subcategory' in node],
        }

        hierarchy_path = osp.join(
            self._save_dir, OpenImagesPath.ANNOTATIONS_DIR, OpenImagesPath.HIERARCHY_NAME)

        with open(hierarchy_path, 'w', encoding='UTF-8') as hierarchy_file:
            json.dump(root_node, hierarchy_file, indent=4)
            hierarchy_file.write('\n')

    def _save_subsets(self):
        # TODO: what if there are no categories?
        label_categories = self._extractor.categories()[AnnotationType.label]

        for subset_name, subset in self._extractor.subsets().items():
            if _RE_INVALID_SUBSET.fullmatch(subset_name):
                raise UnsupportedSubsetNameError(item_id=next(iter(subset)).id, subset=subset)

            image_description_name = f'{subset_name}-images-with-rotation.csv'
            image_description_fields = [
                'ImageID',
                'Subset',
                'OriginalURL',
                'OriginalLandingURL',
                'License',
                'AuthorProfileURL',
                'Author',
                'Title',
                'OriginalSize',
                'OriginalMD5',
                'Thumbnail300KURL',
                'Rotation',
            ]

            label_description_name = f'{subset_name}-annotations-human-imagelabels.csv'
            label_description_fields =  [
                'ImageID',
                'Source',
                'LabelName',
                'Confidence',
            ]

            with \
                self._open_csv_annotation(
                    image_description_name, image_description_fields) as image_description_writer, \
                self._open_csv_annotation(
                    label_description_name, label_description_fields) as label_description_writer \
            :
                image_description_writer.writeheader()
                label_description_writer.writeheader()

                for item in subset:
                    image_description_writer.writerow({
                        'ImageID': item.id, 'Subset': subset_name,
                    })

                    if self._save_images and item.has_image:
                        self._save_image(item, subdir=osp.join('images', subset_name))

                    for annotation in item.annotations:
                        if isinstance(annotation, Label):
                            label_description_writer.writerow({
                                'ImageID': item.id,
                                'LabelName': label_categories[annotation.label].name,
                                'Confidence': str(annotation.attributes.get('score', 1)),
                            })
