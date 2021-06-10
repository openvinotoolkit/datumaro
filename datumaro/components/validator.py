# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from enum import Enum

import numpy as np

from datumaro.components.errors import (MissingAnnotation,
    MultiLabelAnnotations, MissingAttribute, UndefinedLabel,
    UndefinedAttribute, LabelDefinedButNotFound, AttributeDefinedButNotFound,
    OnlyOneLabel, FewSamplesInLabel, FewSamplesInAttribute,
    ImbalancedLabels, ImbalancedAttribute, ImbalancedDistInLabel,
    ImbalancedDistInAttribute, NegativeLength, InvalidValue,
    FarFromLabelMean, FarFromAttrMean, OnlyOneAttributeValue)
from datumaro.components.extractor import AnnotationType, LabelCategories
from datumaro.util import parse_str_enum_value


Severity = Enum('Severity', ['warning', 'error'])

TaskType = Enum('TaskType', ['classification', 'detection', 'segmentation'])


class Validator():
    # statistics templates
    numerical_stat_template = {
        'items_far_from_mean': {},
        'mean': None,
        'stdev': None,
        'min': None,
        'max': None,
        'median': None,
        'histogram': {
            'bins': [],
            'counts': [],
        },
        'distribution': np.array([])
    }

    """
    A base class for task-specific validators.

    Attributes
    ----------
    task_type : str or TaskType
        task type (ie. classification, detection, segmentation)

    Methods
    -------
    compute_statistics(dataset):
        Computes various statistics of the dataset based on task type.
    generate_reports(stats):
        Abstract method that must be implemented in a subclass.
    """

    def __init__(self, task_type, few_samples_thr=None,
            imbalance_ratio_thr=None, far_from_mean_thr=None,
            dominance_ratio_thr=None, topk_bins=None):
        """
        Validator

        Parameters
        ---------------
        few_samples_thr: int
            minimum number of samples per class
            warn user when samples per class is less than threshold
        imbalance_ratio_thr: int
            ratio of majority attribute to minority attribute
            warn user when annotations are unevenly distributed
        far_from_mean_thr: float
            constant used to define mean +/- m * stddev
            warn user when there are too big or small values
        dominance_ratio_thr: float
            ratio of Top-k bin to total
            warn user when dominance ratio is over threshold
        topk_bins: float
            ratio of selected bins with most item number to total bins
            warn user when values are not evenly distributed
        """
        self.task_type = parse_str_enum_value(task_type, TaskType,
            default=TaskType.classification)

        if self.task_type == TaskType.classification:
            self.ann_types = {AnnotationType.label}
            self.str_ann_type = "label"
        elif self.task_type == TaskType.detection:
            self.ann_types = {AnnotationType.bbox}
            self.str_ann_type = "bounding box"
        elif self.task_type == TaskType.segmentation:
            self.ann_types = {AnnotationType.mask, AnnotationType.polygon}
            self.str_ann_type = "mask or polygon"

        self.few_samples_thr = few_samples_thr
        self.imbalance_ratio_thr = imbalance_ratio_thr
        self.far_from_mean_thr = far_from_mean_thr
        self.dominance_thr = dominance_ratio_thr
        self.topk_bins_ratio = topk_bins

    def _compute_common_statistics(self, dataset):
        defined_attr_template = {
            'items_missing_attribute': [],
            'distribution': {}
        }
        undefined_attr_template = {
            'items_with_undefined_attr': [],
            'distribution': {}
        }
        undefined_label_template = {
            'count': 0,
            'items_with_undefined_label': [],
        }

        stats = {
            'label_distribution': {
                'defined_labels': {},
                'undefined_labels': {},
            },
            'attribute_distribution': {
                'defined_attributes': {},
                'undefined_attributes': {}
            },
        }
        stats['total_ann_count'] = 0
        stats['items_missing_annotation'] = []

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_label_dist = label_dist['defined_labels']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        label_categories = dataset.categories().get(AnnotationType.label,
            LabelCategories())
        base_valid_attrs = label_categories.attributes

        for category in label_categories:
            defined_label_dist[category.name] = 0

        filtered_anns = []
        for item in dataset:
            item_key = (item.id, item.subset)
            annotations = []
            for ann in item.annotations:
                if ann.type in self.ann_types:
                    annotations.append(ann)
            ann_count = len(annotations)
            filtered_anns.append((item_key, annotations))

            if ann_count == 0:
                stats['items_missing_annotation'].append(item_key)
            stats['total_ann_count'] += ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label

                    label_stats = undefined_label_dist.setdefault(
                        ann.label, deepcopy(undefined_label_template))
                    label_stats['items_with_undefined_label'].append(
                        item_key)

                    label_stats['count'] += 1
                    valid_attrs = set()
                    missing_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    defined_label_dist[label_name] += 1

                    defined_attr_stats = defined_attr_dist.setdefault(
                        label_name, {})

                    valid_attrs = base_valid_attrs.union(
                        label_categories[ann.label].attributes)
                    ann_attrs = getattr(ann, 'attributes', {}).keys()
                    missing_attrs = valid_attrs.difference(ann_attrs)

                    for attr in valid_attrs:
                        defined_attr_stats.setdefault(
                            attr, deepcopy(defined_attr_template))

                for attr in missing_attrs:
                    attr_dets = defined_attr_stats[attr]
                    attr_dets['items_missing_attribute'].append(
                        item_key)

                for attr, value in ann.attributes.items():
                    if attr not in valid_attrs:
                        undefined_attr_stats = \
                            undefined_attr_dist.setdefault(
                                label_name, {})
                        attr_dets = undefined_attr_stats.setdefault(
                            attr, deepcopy(undefined_attr_template))
                        attr_dets['items_with_undefined_attr'].append(
                            item_key)
                    else:
                        attr_dets = defined_attr_stats[attr]

                    attr_dets['distribution'].setdefault(str(value), 0)
                    attr_dets['distribution'][str(value)] += 1

        return stats, filtered_anns

    @staticmethod
    def _update_prop_distributions(curr_prop_stats, target_stats):
        for prop, val in curr_prop_stats.items():
            prop_stats = target_stats[prop]
            prop_dist = prop_stats['distribution']
            prop_stats['distribution'] = np.append(prop_dist, val)

    @staticmethod
    def _compute_prop_stats_from_dist(dist_by_label, dist_by_attr):
        for label_name, stats in dist_by_label.items():
            prop_stats_list = list(stats.values())
            attr_label = dist_by_attr.get(label_name, {})
            for vals in attr_label.values():
                for val_stats in vals.values():
                    prop_stats_list += list(val_stats.values())

            for prop_stats in prop_stats_list:
                prop_dist = prop_stats.pop('distribution', [])
                if len(prop_dist) > 0:
                    prop_stats['mean'] = np.mean(prop_dist)
                    prop_stats['stdev'] = np.std(prop_dist)
                    prop_stats['min'] = np.min(prop_dist)
                    prop_stats['max'] = np.max(prop_dist)
                    prop_stats['median'] = np.median(prop_dist)

                    counts, bins = np.histogram(prop_dist)
                    prop_stats['histogram']['bins'] = bins.tolist()
                    prop_stats['histogram']['counts'] = counts.tolist()

    def _compute_far_from_mean(self, prop_stats, val, item_key, ann):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        mean = prop_stats['mean']
        stdev = prop_stats['stdev']

        if _far_from_mean(val, mean, stdev):
            items_far_from_mean = prop_stats['items_far_from_mean']
            far_from_mean = items_far_from_mean.setdefault(
                item_key, {})
            far_from_mean[ann.id] = val

    def compute_statistics(self, dataset):
        """
        Computes statistics of the dataset based on task type.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """
        return NotImplementedError

    def _check_missing_label_categories(self, stats):
        validation_reports = []

        if len(stats['label_distribution']['defined_labels']) == 0:
            validation_reports += self._generate_validation_report(
                MissingLabelCategories, Severity.error)

        return validation_reports

    def _check_missing_annotation(self, stats):
        validation_reports = []

        items_missing = stats['items_missing_annotation']
        for item_id, item_subset in items_missing:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset,
                self.str_ann_type)

        return validation_reports

    def _check_missing_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_missing_attr = attr_dets['items_missing_attribute']
        for item_id, item_subset in items_missing_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details)

        return validation_reports

    def _check_undefined_label(self, label_name, label_stats):
        validation_reports = []

        items_with_undefined_label = label_stats['items_with_undefined_label']
        for item_id, item_subset in items_with_undefined_label:
            details = (item_subset, label_name)
            validation_reports += self._generate_validation_report(
                UndefinedLabel, Severity.error, item_id, *details)

        return validation_reports

    def _check_undefined_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_with_undefined_attr = attr_dets['items_with_undefined_attr']
        for item_id, item_subset in items_with_undefined_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details)

        return validation_reports

    def _check_label_defined_but_not_found(self, stats):
        validation_reports = []
        count_by_defined_labels = stats['label_distribution']['defined_labels']
        labels_not_found = [label_name
            for label_name, count in count_by_defined_labels.items()
                if count == 0]

        for label_name in labels_not_found:
            validation_reports += self._generate_validation_report(
                LabelDefinedButNotFound, Severity.warning, label_name)

        return validation_reports

    def _check_attribute_defined_but_not_found(self, label_name, attr_stats):
        validation_reports = []
        attrs_not_found = [attr_name
            for attr_name, attr_dets in attr_stats.items()
                if len(attr_dets['distribution']) == 0]

        for attr_name in attrs_not_found:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                AttributeDefinedButNotFound, Severity.warning, *details)

        return validation_reports

    def _check_only_one_label(self, stats):
        validation_reports = []
        count_by_defined_labels = stats['label_distribution']['defined_labels']
        labels_found = [label_name
            for label_name, count in count_by_defined_labels.items()
                if count > 0]

        if len(labels_found) == 1:
            validation_reports += self._generate_validation_report(
                OnlyOneLabel, Severity.warning, labels_found[0])

        return validation_reports

    def _check_only_one_attribute_value(self, label_name, attr_name, attr_dets):
        validation_reports = []
        values = list(attr_dets['distribution'].keys())

        if len(values) == 1:
            details = (label_name, attr_name, values[0])
            validation_reports += self._generate_validation_report(
                OnlyOneAttributeValue, Severity.warning, *details)

        return validation_reports

    def _check_few_samples_in_label(self, stats):
        validation_reports = []
        thr = self.few_samples_thr

        defined_label_dist = stats['label_distribution']['defined_labels']
        labels_with_few_samples = [(label_name, count)
            for label_name, count in defined_label_dist.items()
                if 0 < count <= thr]

        for label_name, count in labels_with_few_samples:
            validation_reports += self._generate_validation_report(
                FewSamplesInLabel, Severity.warning, label_name, count)

        return validation_reports

    def _check_few_samples_in_attribute(self, label_name,
                                        attr_name, attr_dets):
        validation_reports = []
        thr = self.few_samples_thr

        attr_values_with_few_samples = [(attr_value, count)
            for attr_value, count in attr_dets['distribution'].items()
                if count <= thr]

        for attr_value, count in attr_values_with_few_samples:
            details = (label_name, attr_name, attr_value, count)
            validation_reports += self._generate_validation_report(
                FewSamplesInAttribute, Severity.warning, *details)

        return validation_reports

    def _check_imbalanced_labels(self, stats):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        defined_label_dist = stats['label_distribution']['defined_labels']
        count_by_defined_labels = [count
            for label, count in defined_label_dist.items()]

        if len(count_by_defined_labels) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_labels)
        count_min = np.min(count_by_defined_labels)
        balance = count_max / count_min if count_min > 0 else float('inf')
        if balance >= thr:
            validation_reports += self._generate_validation_report(
                ImbalancedLabels, Severity.warning)

        return validation_reports

    def _check_imbalanced_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        count_by_defined_attr = list(attr_dets['distribution'].values())
        if len(count_by_defined_attr) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_attr)
        count_min = np.min(count_by_defined_attr)
        balance = count_max / count_min if count_min > 0 else float('inf')
        if balance >= thr:
            validation_reports += self._generate_validation_report(
                ImbalancedAttribute, Severity.warning, label_name, attr_name)

        return validation_reports

    def _check_imbalanced_dist_in_label(self, label_name, label_stats):
        validation_reports = []
        thr = self.dominance_thr
        topk_ratio = self.topk_bins_ratio

        for prop, prop_stats in label_stats.items():
            value_counts = prop_stats['histogram']['counts']
            n_bucket = len(value_counts)
            if n_bucket < 2:
                continue
            topk = max(1, int(np.around(n_bucket * topk_ratio)))

            if topk > 0:
                topk_values = np.sort(value_counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(value_counts)
                if ratio >= thr:
                    details = (label_name, f"{self.str_ann_type} {prop}")
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInLabel, Severity.warning, *details)

        return validation_reports

    def _check_imbalanced_dist_in_attr(self, label_name, attr_name, attr_stats):
        validation_reports = []
        thr = self.dominance_thr
        topk_ratio = self.topk_bins_ratio

        for attr_value, value_stats in attr_stats.items():
            for prop, prop_stats in value_stats.items():
                value_counts = prop_stats['histogram']['counts']
                n_bucket = len(value_counts)
                if n_bucket < 2:
                    continue
                topk = max(1, int(np.around(n_bucket * topk_ratio)))

                if topk > 0:
                    topk_values = np.sort(value_counts)[-topk:]
                    ratio = np.sum(topk_values) / np.sum(value_counts)
                    if ratio >= thr:
                        details = (label_name, attr_name, attr_value,
                                   f"{self.str_ann_type} {prop}")
                        validation_reports += self._generate_validation_report(
                            ImbalancedDistInAttribute,
                            Severity.warning,
                            *details
                        )

        return validation_reports

    def _check_invalid_value(self, stats):
        validation_reports = []

        items_w_invalid_val = stats['items_with_invalid_value']
        for item_dets, anns_w_invalid_val in items_w_invalid_val.items():
            item_id, item_subset = item_dets
            for ann_id, props in anns_w_invalid_val.items():
                for prop in props:
                    details = (item_subset, ann_id,
                               f"{self.str_ann_type} {prop}")
                    validation_reports += self._generate_validation_report(
                        InvalidValue, Severity.error, item_id, *details)

        return validation_reports

    def _check_far_from_label_mean(self, label_name, label_stats):
        validation_reports = []

        for prop, prop_stats in label_stats.items():
            items_far_from_mean = prop_stats['items_far_from_mean']
            if prop_stats['mean'] is not None:
                mean = round(prop_stats['mean'], 2)

            for item_dets, anns_far in items_far_from_mean.items():
                item_id, item_subset = item_dets
                for ann_id, val in anns_far.items():
                    val = round(val, 2)
                    details = (item_subset, label_name, ann_id,
                               f"{self.str_ann_type} {prop}", mean, val)
                    validation_reports += self._generate_validation_report(
                        FarFromLabelMean, Severity.warning, item_id, *details)

        return validation_reports

    def _check_far_from_attr_mean(self, label_name, attr_name, attr_stats):
        validation_reports = []

        for attr_value, value_stats in attr_stats.items():
            for prop, prop_stats in value_stats.items():
                items_far_from_mean = prop_stats['items_far_from_mean']
                if prop_stats['mean'] is not None:
                    mean = round(prop_stats['mean'], 2)

                for item_dets, anns_far in items_far_from_mean.items():
                    item_id, item_subset = item_dets
                    for ann_id, val in anns_far.items():
                        val = round(val, 2)
                        details = (item_subset, label_name, ann_id, attr_name,
                                   attr_value, f"{self.str_ann_type} {prop}",
                                   mean, val)
                        validation_reports += self._generate_validation_report(
                            FarFromAttrMean,
                            Severity.warning,
                            item_id,
                            *details
                        )

        return validation_reports

    def generate_reports(self, stats):
        raise NotImplementedError('Should be implemented in a subclass.')

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]