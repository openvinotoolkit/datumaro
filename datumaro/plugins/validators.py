# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy

import numpy as np

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import (
    AttributeDefinedButNotFound, FarFromAttrMean, FarFromLabelMean,
    FewSamplesInAttribute, FewSamplesInLabel, ImbalancedAttribute,
    ImbalancedDistInAttribute, ImbalancedDistInLabel, ImbalancedLabels,
    InvalidValue, LabelDefinedButNotFound, MissingAnnotation, MissingAttribute,
    MissingLabelCategories, MultiLabelAnnotations, NegativeLength,
    OnlyOneAttributeValue, OnlyOneLabel, UndefinedAttribute, UndefinedLabel,
)
from datumaro.components.extractor import AnnotationType, LabelCategories
from datumaro.components.validator import Severity, TaskType, Validator
from datumaro.util import parse_str_enum_value


class _TaskValidator(Validator, CliPlugin):
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
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument('-fs', '--few-samples-thr',
            default=1, type=int,
            help="Threshold for giving a warning for minimum number of "
                "samples per class (default: %(default)s)")
        parser.add_argument('-ir', '--imbalance-ratio-thr',
            default=50, type=int,
            help="Threshold for giving data imbalance warning. "
                "IR(imbalance ratio) = majority/minority "
                "(default: %(default)s)")
        parser.add_argument('-m', '--far-from-mean-thr',
            default=5.0, type=float,
            help="Threshold for giving a warning that data is far from mean. "
                "A constant used to define mean +/- k * standard deviation "
                "(default: %(default)s)")
        parser.add_argument('-dr', '--dominance-ratio-thr',
            default=0.8, type=float,
            help="Threshold for giving a warning for bounding box imbalance. "
                "Dominace_ratio = ratio of Top-k bin to total in histogram "
                "(default: %(default)s)")
        parser.add_argument('-k', '--topk-bins', default=0.1, type=float,
            help="Ratio of bins with the highest number of data"
                "to total bins in the histogram. A value in the range [0, 1] "
                "(default: %(default)s)")
        return parser

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

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]


class ClassificationValidator(_TaskValidator):
    """
    A specific validator class for classification task.
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


class DetectionValidator(_TaskValidator):
    """
    A specific validator class for detection task.
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


class SegmentationValidator(_TaskValidator):
    """
    A specific validator class for (instance) segmentation task.
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
