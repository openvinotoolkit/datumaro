# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import shutil
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile

from datumaro import AnnotationType, Dataset, HLOps, LabelCategories
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.filter import DatasetItemEncoder
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.plugins.validators import (
    ClassificationValidator,
    DetectionValidator,
    SegmentationValidator,
)


class DataRepo:
    """
    Implements the data repo for Dataset Projects
    """

    def __init__(self):
        self._root_path = ".data_repo"
        self._dataset_dir = "dataset"

    def get_dataset_dir(_self, file_id: str) -> str:
        """
        Get directory of unzipped dataset.

        :param file_id: fild_id of uploaded zip file
        :return: path to dataset directory
        """
        directory = os.path.join(_self._root_path, file_id, _self._dataset_dir)
        os.makedirs(directory, exist_ok=True)
        return directory

    def save_file(_self, uploaded_file: UploadedFile) -> str:
        directory = os.path.join(_self._root_path, uploaded_file.file_id)
        path = os.path.join(directory, uploaded_file.name)
        os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path

    def delete_by_id(_self, fild_id: str):
        """
        Delete (unzipped) dataset for a given file_id

        :param file_id: fild_id of uploaded zip file
        :raises: FileNotFoundError
        """
        path = os.path.join(_self._root_path, str(fild_id))
        if os.path.exists(path):
            shutil.rmtree(path)


class DatasetHelper:
    """
    Import dm_dataset from DataRepo
    """

    def __init__(self, dataset_root: str = None):
        self._dataset_dir = dataset_root
        self._detected_formats = None
        self._dm_dataset = None
        self._format = ""
        self._init_dependent_variables()

    def __del__(self):
        print(self._dataset_dir)
        file_id = os.path.basename(os.path.dirname(self._dataset_dir))
        DataRepo().delete_by_id(file_id)

    def _init_dependent_variables(self):
        # when dataset is updated, some variables should be initialized.
        self._val_reports = {}
        self._image_stats = None
        self._ann_stats = None
        self._image_size_info = None
        self._xml_items = {}
        self._subset_to_ids = {}

    def detect_format(self) -> List[str]:
        if self._detected_formats is None:
            self._detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(path=self._dataset_dir)
        return self._detected_formats

    def import_dataset(self, format) -> Dataset:
        if format != self._format:
            self._format = format
            self._dm_dataset = Dataset.import_from(path=self._dataset_dir, format=self._format)
            self._init_dependent_variables()
        return self._dm_dataset

    def update_dataset(self, dataset):
        return NotImplementedError()

    def dataset(self) -> Dataset:
        return self._dm_dataset

    def format(self) -> str:
        return self._format

    def subset_to_ids(self) -> Dict[str, list]:
        if not self._subset_to_ids and self._dm_dataset:
            keys = defaultdict(list)
            for item in self._dm_dataset:
                keys[item.subset].append(item.id)
            self._subset_to_ids = keys
        if self._dm_dataset and self._subset_to_ids.keys() != self._dm_dataset.subsets().keys():
            keys = defaultdict(list)
            for item in self._dm_dataset:
                keys[item.subset].append(item.id)
            self._subset_to_ids = keys
        return self._subset_to_ids

    def validate(self, task: str):
        if task not in self._val_reports:
            validators = {
                "classification": ClassificationValidator,
                "detection": DetectionValidator,
                "segmentation": SegmentationValidator,
            }
            validator = validators.get(task.lower(), ClassificationValidator)()
            reports = validator.validate(self._dm_dataset)
            self._val_reports[task] = reports
        return self._val_reports[task]

    def aggregate(self, from_subsets, to_subset) -> Dataset:
        self._dm_dataset = HLOps.aggregate(
            self._dm_dataset, from_subsets=from_subsets, to_subset=to_subset
        )
        self._val_reports = {}
        return self._dm_dataset

    def transform(self, method: str, **kwargs):
        self._dm_dataset = self._dm_dataset.transform(method, **kwargs)
        self._init_dependent_variables()
        return self._dm_dataset

    def filter(self, expr: str, filter_args):
        self._dm_dataset = self._dm_dataset.filter(expr, **filter_args)
        self._init_dependent_variables()
        return self._dm_dataset

    def get_xml(self, subset: str, id: str) -> Union[str, None]:
        key = (subset, id)
        xml_item = self._xml_items.get(key, None)
        if xml_item is None and self._dm_dataset is not None:
            item = self._dm_dataset.get(id, subset)
            encoded_item = DatasetItemEncoder.encode(item, self._dm_dataset.categories())
            xml_item = DatasetItemEncoder.to_string(encoded_item)
            self._xml_items[key] = xml_item
        return xml_item

    def export(self, save_dir: str, format: str, **kwargs):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._dm_dataset.export(save_dir=save_dir, format=format, **kwargs)

    def merge(self, source_datasets, merge_policy, report_path=None, **kwargs):
        return NotImplementedError()

    def get_image_stats(self, force_init: bool = False):
        return NotImplementedError()

    def get_ann_stats(self, force_init: bool = False):
        return NotImplementedError()

    def get_image_size_info(self):
        return NotImplementedError()

    @property
    def num_labels(self):
        try:
            return len(self._dm_dataset.categories().get(AnnotationType.label, LabelCategories()))
        except Exception:
            return 0


class SingleDatasetHelper(DatasetHelper):
    def get_image_stats(self, force_init: bool = False):
        if not self._image_stats or force_init:
            self._image_stats = compute_image_statistics(self._dm_dataset)
        return self._image_stats

    def get_ann_stats(self, force_init: bool = False):
        if not self._ann_stats or force_init:
            self._ann_stats = compute_ann_statistics(self._dm_dataset)
        return self._ann_stats

    def get_image_size_info(self):
        if not self._image_size_info:
            labels = self._dm_dataset.categories().get(AnnotationType.label, LabelCategories())

            def get_label(ann):
                return labels.items[ann.label].name if ann.label is not None else None

            all_sizes = []
            by_subsets = defaultdict(list)
            by_labels = defaultdict(list)
            for item in self._dm_dataset:
                if item.media:
                    size = item.media.as_dict().get("size", None)
                    if size:
                        size_info = {"x": size[1], "y": size[0]}
                        all_sizes.append(size)
                        by_subsets[item.subset].append(size_info)
                        for (
                            ann
                        ) in (
                            item.annotations
                        ):  # size can be duplicated because item can have multiple annotations
                            try:
                                label = get_label(ann)
                                if label:
                                    by_labels[label].append(size_info)
                            except Exception:
                                pass

            mean = np.mean(all_sizes, axis=0) if all_sizes else [0, 0]
            std = np.std(all_sizes, axis=0) if all_sizes else [0, 0]

            self._image_size_info = {
                "by_subsets": by_subsets,
                "by_labels": by_labels,
                "image_size": {"mean": mean, "std": std},
            }

        return self._image_size_info


class MultipleDatasetHelper(DatasetHelper):
    def get_ann_stats(self, force_init: bool = False):
        if not self._ann_stats or force_init:
            self._ann_stats = compute_ann_statistics(self._dm_dataset)
        return self._ann_stats

    def update_dataset(self, dataset):
        self._dm_dataset = dataset

    def merge(self, source_datasets, merge_policy, report_path=None, **kwargs):
        merged_dataset = HLOps.merge(
            *source_datasets, merge_policy=merge_policy, report_path=report_path, **kwargs
        )
        return merged_dataset
