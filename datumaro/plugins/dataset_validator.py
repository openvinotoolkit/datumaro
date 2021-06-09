# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import numpy as np
from copy import deepcopy
from enum import Enum

from datumaro.components.validator import (Severity, TaskType, Validator)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset import IDataset
from datumaro.components.errors import MultiLabelAnnotations
from datumaro.components.extractor import AnnotationType, LabelCategories
from datumaro.util import parse_str_enum_value

from datumaro.cli.util import MultilineFormatter
from datumaro.cli.util.project import generate_next_file_name, load_project

def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="Validate project",
        description="""
            Validates project based on specified task type and stores
            results like statistics, reports and summary in JSON file.
        """,
        formatter_class=MultilineFormatter)
    parser.add_argument('-t', '--task_type',
        choices=[task_type.name for task_type in TaskType],
        help="Task type for validation")
    parser.add_argument('-s', '--subset', dest='subset_name', default=None,
        help="Subset to validate (default: None)")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to validate (default: current dir)")
    parser.add_argument('-fs', '--few_samples_thr', default=1, type=int,
        help="Threshold for giving a warning for minimum number of"
             "samples per class")
    parser.add_argument('-ir', '--imbalance_ratio_thr', default=50, type=int,
        help="Threshold for giving data imbalance warning;"
             "IR(imbalance ratio) = majority/minority")
    parser.add_argument('-m', '--far_from_mean_thr', default=5.0, type=float,
        help="Threshold for giving a warning that data is far from mean;"
             "A constant used to define mean +/- k * standard deviation;")
    parser.add_argument('-dr', '--dominance_ratio_thr', default=0.8, type=float,
        help="Threshold for giving a warning for bounding box imbalance;"
            "Dominace_ratio = ratio of Top-k bin to total in histogram;")
    parser.add_argument('-k', '--topk_bins', default=0.1, type=float,
        help="Ratio of bins with the highest number of data"
             "to total bins in the histogram; [0, 1]; 0.1 = 10%;")
    parser.set_defaults(command=validate_command)
    return parser

def validate_command(args):
    project = load_project(args.project_dir)
    task_type = args.task_type
    subset_name = args.subset_name
    dst_file_name = f'validation_results-{task_type}'

    dataset = project.make_dataset()
    if subset_name is not None:
        dataset = dataset.get_subset(subset_name)
        dst_file_name += f'-{subset_name}'

    dataset_validator = project.env.validators['dataset'](task_type, args)

    validation_results = dataset_validator.validate_annotations(dataset)

    def numpy_encoder(obj):
        if isinstance(obj, np.generic):
            return obj.item()

    def _make_serializable(d):
        for key, val in list(d.items()):
            # tuple key to str
            if isinstance(key, tuple):
                d[str(key)] = val
                d.pop(key)
            if isinstance(val, dict):
                _make_serializable(val)

    _make_serializable(validation_results)

    dst_file = generate_next_file_name(dst_file_name, ext='.json')
    log.info("Writing project validation results to '%s'" % dst_file)
    with open(dst_file, 'w') as f:
        json.dump(validation_results, f, indent=4, sort_keys=True,
                  default=numpy_encoder)

class ClassificationValidator(Validator):
    """
    A validator class for classification tasks.
    """

    def __init__(self, few_samples_thr, imbalance_ratio_thr,
            far_from_mean_thr, dominance_ratio_thr, topk_bins):
        super().__init__(task_type=TaskType.classification,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr, topk_bins=topk_bins)

    def _check_multi_label_annotations(self, stats):
        validation_reports = []

        items_with_multiple_labels = stats['items_with_multiple_labels']
        for item_id, item_subset in items_with_multiple_labels:
            validation_reports += self._generate_validation_report(
                MultiLabelAnnotations, Severity.error, item_id, item_subset)

        return validation_reports

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the classification task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_anns = self._compute_common_statistics(dataset)

        stats['items_with_multiple_labels'] = []

        for item_key, anns in filtered_anns:
            ann_count = len(anns)
            if ann_count > 1:
                stats['items_with_multiple_labels'].append(item_key)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for classification tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats: Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = []

        reports += self._check_missing_label_categories(stats)
        reports += self._check_missing_annotation(stats)
        reports += self._check_multi_label_annotations(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(
                label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_few_samples_in_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_imbalanced_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_only_one_attribute_value(
                    label_name, attr_name, attr_dets)
                reports += self._check_missing_attribute(
                    label_name, attr_name, attr_dets)

        for label_name, label_stats in undefined_label_dist.items():
            reports += self._check_undefined_label(label_name, label_stats)

        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(
                    label_name, attr_name, attr_dets)

        return reports

class DetectionValidator(Validator):
    """
    A validator class for detection tasks.
    """
    def __init__(self, few_samples_thr, imbalance_ratio_thr,
            far_from_mean_thr, dominance_ratio_thr, topk_bins):
        super().__init__(task_type=TaskType.detection,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr, topk_bins=topk_bins)

    def _check_negative_length(self, stats):
        validation_reports = []

        items_w_neg_len = stats['items_with_negative_length']
        for item_dets, anns_w_neg_len in items_w_neg_len.items():
            item_id, item_subset = item_dets
            for ann_id, props in anns_w_neg_len.items():
                for prop, val in props.items():
                    val = round(val, 2)
                    details = (item_subset, ann_id,
                               f"{self.str_ann_type} {prop}", val)
                    validation_reports += self._generate_validation_report(
                        NegativeLength, Severity.error, item_id, *details)

        return validation_reports

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the detection task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_anns = self._compute_common_statistics(dataset)

        # detection-specific
        bbox_template = {
            'width': deepcopy(self.numerical_stat_template),
            'height': deepcopy(self.numerical_stat_template),
            'area(wxh)': deepcopy(self.numerical_stat_template),
            'ratio(w/h)': deepcopy(self.numerical_stat_template),
            'short': deepcopy(self.numerical_stat_template),
            'long': deepcopy(self.numerical_stat_template)
        }

        stats['items_with_negative_length'] = {}
        stats['items_with_invalid_value'] = {}
        stats['bbox_distribution_in_label'] = {}
        stats['bbox_distribution_in_attribute'] = {}
        stats['bbox_distribution_in_dataset_item'] = {}

        dist_by_label = stats['bbox_distribution_in_label']
        dist_by_attr = stats['bbox_distribution_in_attribute']
        bbox_dist_in_item = stats['bbox_distribution_in_dataset_item']
        items_w_neg_len = stats['items_with_negative_length']
        items_w_invalid_val = stats['items_with_invalid_value']

        def _generate_ann_bbox_info(_x, _y, _w, _h, area,
                                    ratio, _short, _long):
            return {
                'x': _x,
                'y': _y,
                'width': _w,
                'height': _h,
                'area(wxh)': area,
                'ratio(w/h)': ratio,
                'short': _short,
                'long': _long,
            }

        def _update_bbox_stats_by_label(item_key, ann, bbox_label_stats):
            bbox_has_error = False

            _x, _y, _w, _h = ann.get_bbox()
            area = ann.get_area()

            if _h != 0 and _h != float('inf'):
                ratio = _w / _h
            else:
                ratio = float('nan')

            _short = _w if _w < _h else _h
            _long = _w if _w > _h else _h

            ann_bbox_info = _generate_ann_bbox_info(
                _x, _y, _w, _h, area, ratio, _short, _long)

            for prop, val in ann_bbox_info.items():
                if val == float('inf') or np.isnan(val):
                    bbox_has_error = True
                    anns_w_invalid_val = items_w_invalid_val.setdefault(
                        item_key, {})
                    invalid_props = anns_w_invalid_val.setdefault(
                        ann.id, [])
                    invalid_props.append(prop)

            for prop in ['width', 'height']:
                val = ann_bbox_info[prop]
                if val < 1:
                    bbox_has_error = True
                    anns_w_neg_len = items_w_neg_len.setdefault(
                        item_key, {})
                    neg_props = anns_w_neg_len.setdefault(ann.id, {})
                    neg_props[prop] = val

            if not bbox_has_error:
                ann_bbox_info.pop('x')
                ann_bbox_info.pop('y')
                self._update_prop_distributions(ann_bbox_info, bbox_label_stats)

            return ann_bbox_info, bbox_has_error

        label_categories = dataset.categories().get(AnnotationType.label,
            LabelCategories())
        base_valid_attrs = label_categories.attributes

        for item_key, annotations in filtered_anns:
            ann_count = len(annotations)

            bbox_dist_in_item[item_key] = ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label
                    valid_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    valid_attrs = base_valid_attrs.union(
                        label_categories[ann.label].attributes)

                    bbox_label_stats = dist_by_label.setdefault(
                        label_name, deepcopy(bbox_template))
                    ann_bbox_info, bbox_has_error = \
                        _update_bbox_stats_by_label(
                            item_key, ann, bbox_label_stats)

                for attr, value in ann.attributes.items():
                    if attr in valid_attrs:
                        bbox_attr_label = dist_by_attr.setdefault(
                            label_name, {})
                        bbox_attr_stats = bbox_attr_label.setdefault(
                            attr, {})
                        bbox_val_stats = bbox_attr_stats.setdefault(
                            str(value), deepcopy(bbox_template))

                        if not bbox_has_error:
                            self._update_prop_distributions(
                                ann_bbox_info, bbox_val_stats)

        # Compute prop stats from distribution
        self._compute_prop_stats_from_dist(dist_by_label, dist_by_attr)

        def _is_valid_ann(item_key, ann):
            has_defined_label = 0 <= ann.label < len(label_categories)
            if not has_defined_label:
                return False

            bbox_has_neg_len = ann.id in items_w_neg_len.get(
                item_key, {})
            bbox_has_invalid_val = ann.id in items_w_invalid_val.get(
                item_key, {})
            return not (bbox_has_neg_len or bbox_has_invalid_val)

        def _update_props_far_from_mean(item_key, ann):
            valid_attrs = base_valid_attrs.union(
                label_categories[ann.label].attributes)
            label_name = label_categories[ann.label].name
            bbox_label_stats = dist_by_label[label_name]

            _x, _y, _w, _h = ann.get_bbox()
            area = ann.get_area()
            ratio = _w / _h
            _short = _w if _w < _h else _h
            _long = _w if _w > _h else _h

            ann_bbox_info = _generate_ann_bbox_info(
                _x, _y, _w, _h, area, ratio, _short, _long)
            ann_bbox_info.pop('x')
            ann_bbox_info.pop('y')

            for prop, val in ann_bbox_info.items():
                prop_stats = bbox_label_stats[prop]
                self._compute_far_from_mean(prop_stats, val, item_key, ann)

            for attr, value in ann.attributes.items():
                if attr in valid_attrs:
                    bbox_attr_stats = dist_by_attr[label_name][attr]
                    bbox_val_stats = bbox_attr_stats[str(value)]

                    for prop, val in ann_bbox_info.items():
                        prop_stats = bbox_val_stats[prop]
                        self._compute_far_from_mean(prop_stats, val,
                                                    item_key, ann)

        for item_key, annotations in filtered_anns:
            for ann in annotations:
                if _is_valid_ann(item_key, ann):
                    _update_props_far_from_mean(item_key, ann)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for detection tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats : Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = []

        reports += self._check_missing_label_categories(stats)
        reports += self._check_missing_annotation(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)
        reports += self._check_negative_length(stats)
        reports += self._check_invalid_value(stats)

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        dist_by_label = stats['bbox_distribution_in_label']
        dist_by_attr = stats['bbox_distribution_in_attribute']

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(
                label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_few_samples_in_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_imbalanced_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_only_one_attribute_value(
                    label_name, attr_name, attr_dets)
                reports += self._check_missing_attribute(
                    label_name, attr_name, attr_dets)

            bbox_label_stats = dist_by_label[label_name]
            bbox_attr_label = dist_by_attr.get(label_name, {})

            reports += self._check_far_from_label_mean(
                label_name, bbox_label_stats)
            reports += self._check_imbalanced_dist_in_label(
                label_name, bbox_label_stats)

            for attr_name, bbox_attr_stats in bbox_attr_label.items():
                reports += self._check_far_from_attr_mean(
                    label_name, attr_name, bbox_attr_stats)
                reports += self._check_imbalanced_dist_in_attr(
                    label_name, attr_name, bbox_attr_stats)

        for label_name, label_stats in undefined_label_dist.items():
            reports += self._check_undefined_label(label_name, label_stats)

        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(
                    label_name, attr_name, attr_dets)

        return reports

class SegmentationValidator(Validator):
    """
    A validator class for (instance) segmentation tasks.
    """

    def __init__(self, few_samples_thr, imbalance_ratio_thr,
            far_from_mean_thr, dominance_ratio_thr, topk_bins):
        super().__init__(task_type=TaskType.segmentation,
            few_samples_thr=few_samples_thr,
            imbalance_ratio_thr=imbalance_ratio_thr,
            far_from_mean_thr=far_from_mean_thr,
            dominance_ratio_thr=dominance_ratio_thr, topk_bins=topk_bins)

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset for the segmentation task.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

        stats, filtered_anns = self._compute_common_statistics(dataset)

        # segmentation-specific
        mask_template = {
            'area': deepcopy(self.numerical_stat_template),
            'width': deepcopy(self.numerical_stat_template),
            'height': deepcopy(self.numerical_stat_template)
        }

        stats['items_with_invalid_value'] = {}
        stats['mask_distribution_in_label'] = {}
        stats['mask_distribution_in_attribute'] = {}
        stats['mask_distribution_in_dataset_item'] = {}

        dist_by_label = stats['mask_distribution_in_label']
        dist_by_attr = stats['mask_distribution_in_attribute']
        mask_dist_in_item = stats['mask_distribution_in_dataset_item']
        items_w_invalid_val = stats['items_with_invalid_value']

        def _generate_ann_mask_info(area, _w, _h):
            return {
                'area': area,
                'width': _w,
                'height': _h,
            }

        def _update_mask_stats_by_label(item_key, ann, mask_label_stats):
            mask_has_error = False

            _x, _y, _w, _h = ann.get_bbox()

            # Detete the following block when #226 is resolved
            # https://github.com/openvinotoolkit/datumaro/issues/226
            if ann.type == AnnotationType.mask:
                _w += 1
                _h += 1

            area = ann.get_area()

            ann_mask_info = _generate_ann_mask_info(area, _w, _h)

            for prop, val in ann_mask_info.items():
                if val == float('inf') or np.isnan(val):
                    mask_has_error = True
                    anns_w_invalid_val = items_w_invalid_val.setdefault(
                        item_key, {})
                    invalid_props = anns_w_invalid_val.setdefault(
                        ann.id, [])
                    invalid_props.append(prop)

            if not mask_has_error:
                self._update_prop_distributions(ann_mask_info, mask_label_stats)

            return ann_mask_info, mask_has_error

        label_categories = dataset.categories().get(AnnotationType.label,
            LabelCategories())
        base_valid_attrs = label_categories.attributes

        for item_key, annotations in filtered_anns:
            ann_count = len(annotations)
            mask_dist_in_item[item_key] = ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label
                    valid_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    valid_attrs = base_valid_attrs.union(
                        label_categories[ann.label].attributes)

                    mask_label_stats = dist_by_label.setdefault(
                        label_name, deepcopy(mask_template))
                    ann_mask_info, mask_has_error = \
                        _update_mask_stats_by_label(
                            item_key, ann, mask_label_stats)

                for attr, value in ann.attributes.items():
                    if attr in valid_attrs:
                        mask_attr_label = dist_by_attr.setdefault(
                            label_name, {})
                        mask_attr_stats = mask_attr_label.setdefault(
                            attr, {})
                        mask_val_stats = mask_attr_stats.setdefault(
                            str(value), deepcopy(mask_template))

                        if not mask_has_error:
                            self._update_prop_distributions(
                                ann_mask_info, mask_val_stats)

        # compute prop stats from dist.
        self._compute_prop_stats_from_dist(dist_by_label, dist_by_attr)

        def _is_valid_ann(item_key, ann):
            has_defined_label = 0 <= ann.label < len(label_categories)
            if not has_defined_label:
                return False

            mask_has_invalid_val = ann.id in items_w_invalid_val.get(
                item_key, {})
            return not mask_has_invalid_val

        def _update_props_far_from_mean(item_key, ann):
            valid_attrs = base_valid_attrs.union(
                label_categories[ann.label].attributes)
            label_name = label_categories[ann.label].name
            mask_label_stats = dist_by_label[label_name]

            _x, _y, _w, _h = ann.get_bbox()

            # Detete the following block when #226 is resolved
            # https://github.com/openvinotoolkit/datumaro/issues/226
            if ann.type == AnnotationType.mask:
                _w += 1
                _h += 1
            area = ann.get_area()

            ann_mask_info = _generate_ann_mask_info(area, _w, _h)

            for prop, val in ann_mask_info.items():
                prop_stats = mask_label_stats[prop]
                self._compute_far_from_mean(prop_stats, val, item_key, ann)

            for attr, value in ann.attributes.items():
                if attr in valid_attrs:
                    mask_attr_stats = dist_by_attr[label_name][attr]
                    mask_val_stats = mask_attr_stats[str(value)]

                    for prop, val in ann_mask_info.items():
                        prop_stats = mask_val_stats[prop]
                        self._compute_far_from_mean(prop_stats, val,
                                                    item_key, ann)

        for item_key, annotations in filtered_anns:
            for ann in annotations:
                if _is_valid_ann(item_key, ann):
                    _update_props_far_from_mean(item_key, ann)

        return stats

    def generate_reports(self, stats):
        """
        Validates the dataset for segmentation tasks based on its statistics.

        Parameters
        ----------
        dataset : IDataset object
        stats : Dict object

        Returns
        -------
        reports (list): List of validation reports (DatasetValidationError).
        """

        reports = []

        reports += self._check_missing_label_categories(stats)
        reports += self._check_missing_annotation(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)
        reports += self._check_invalid_value(stats)

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        dist_by_label = stats['mask_distribution_in_label']
        dist_by_attr = stats['mask_distribution_in_attribute']

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(
                label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_few_samples_in_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_imbalanced_attribute(
                    label_name, attr_name, attr_dets)
                reports += self._check_only_one_attribute_value(
                    label_name, attr_name, attr_dets)
                reports += self._check_missing_attribute(
                    label_name, attr_name, attr_dets)

            mask_label_stats = dist_by_label[label_name]
            mask_attr_label = dist_by_attr.get(label_name, {})

            reports += self._check_far_from_label_mean(
                label_name, mask_label_stats)
            reports += self._check_imbalanced_dist_in_label(
                label_name, mask_label_stats)

            for attr_name, mask_attr_stats in mask_attr_label.items():
                reports += self._check_far_from_attr_mean(
                    label_name, attr_name, mask_attr_stats)
                reports += self._check_imbalanced_dist_in_attr(
                    label_name, attr_name, mask_attr_stats)

        for label_name, label_stats in undefined_label_dist.items():
            reports += self._check_undefined_label(label_name, label_stats)

        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(
                    label_name, attr_name, attr_dets)

        return reports

class DatasetValidator(Validator, CliPlugin):

    def __init__(self, task_type, args):
        self.task_type = parse_str_enum_value(task_type, TaskType)
        self.few_samples_thr = args.few_samples_thr
        self.imbalance_ratio_thr = args.imbalance_ratio_thr
        self.far_from_mean_thr = args.far_from_mean_thr
        self.dominance_ratio_thr = args.dominance_ratio_thr
        self.topk_bins = args.topk_bins

    def validate_annotations(self, dataset: IDataset):
        """
        Returns the validation results of a dataset based on task type.
        Args:
            dataset (IDataset): Dataset to be validated
            task_type (str or TaskType): Type of the task
                (classification, detection, segmentation)
        Raises:
            ValueError
        Returns:
            validation_results (dict):
                Dict with validation statistics, reports and summary.
        """

        validation_results = {}

        if self.task_type == TaskType.classification:
            validator = ClassificationValidator(few_samples_thr=self.few_samples_thr,
                imbalance_ratio_thr=self.imbalance_ratio_thr,
                far_from_mean_thr=self.far_from_mean_thr,
                dominance_ratio_thr=self.dominance_ratio_thr,
                topk_bins=self.topk_bins)
        elif self.task_type == TaskType.detection:
            validator = DetectionValidator(few_samples_thr=self.few_samples_thr,
                imbalance_ratio_thr=self.imbalance_ratio_thr,
                far_from_mean_thr=self.far_from_mean_thr,
                dominance_ratio_thr=self.dominance_ratio_thr,
                topk_bins=self.topk_bins)
        elif self.task_type == TaskType.segmentation:
            validator = SegmentationValidator(few_samples_thr=self.few_samples_thr,
                imbalance_ratio_thr=self.imbalance_ratio_thr,
                far_from_mean_thr=self.far_from_mean_thr,
                dominance_ratio_thr=self.dominance_ratio_thr,
                topk_bins=self.topk_bins)

        if not isinstance(dataset, IDataset):
            raise TypeError("Invalid dataset type '%s'" % type(dataset))

        # generate statistics
        stats = validator.compute_statistics(dataset)
        validation_results['statistics'] = stats

        # generate validation reports and summary
        reports = validator.generate_reports(stats)
        reports = list(map(lambda r: r.to_dict(), reports))

        summary = {
            'errors': sum(map(lambda r: r['severity'] == 'error', reports)),
            'warnings': sum(map(lambda r: r['severity'] == 'warning', reports))
        }

        validation_results['validation_reports'] = reports
        validation_results['summary'] = summary

        return validation_results