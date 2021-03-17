# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
from enum import Enum
import numpy as np

from datumaro.components.dataset import IDataset
from datumaro.components.errors import (MissingLabelCategories,
    MissingLabelAnnotation, MultiLabelAnnotations, MissingAttribute,
    UndefinedLabel, UndefinedAttribute, LabelDefinedButNotFound,
    AttributeDefinedButNotFound, OnlyOneLabel, FewSamplesInLabel,
    FewSamplesInAttribute, ImbalancedLabels, ImbalancedAttribute,
    ImbalancedBboxDistInLabel, ImbalancedBboxDistInAttribute,
    MissingBboxAnnotation, NegativeLength, InvalidValue, FarFromLabelMean,
    FarFromAttrMean, OnlyOneAttributeValue)
from datumaro.components.extractor import AnnotationType


Severity = Enum('Severity',
    [
        'warning',
        'error'
    ]
)


class _Validator:
    """
    A base class for task-specific validators.

    ...

    Attributes
    ----------
    task_type : str
        task type (ie. 'classification', etc.) (default is 'classification')
    ann_type : AnnotationType
        annotation type to validate (default is AnnotationType.label)
    far_from_mean_thr : float
        constant used to define mean +/- k * stdev (default is None)

    Methods
    -------
    compute_statistics(dataset):
        Computes various statistics of the dataset based on task type.
    generate_reports(stats):
        Abstract method that must be implemented in a subclass.
    """

    def __init__(self,
                task_type='classification',
                ann_type=AnnotationType.label,
                far_from_mean_thr=None):
        self.task_type = task_type
        self.ann_type = ann_type
        self.far_from_mean_thr = far_from_mean_thr

    def compute_statistics(self, dataset):
        """
        Computes various statistics of the dataset based on task type.

        Parameters
        ----------
        dataset : IDataset object

        Returns
        -------
        stats (dict): A dict object containing statistics of the dataset.
        """

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

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_label_dist = label_dist['defined_labels']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        label_categories = dataset.categories().get(AnnotationType.label)
        if label_categories is None:
            label_categories = []

        if self.task_type == 'classification':
            stats['total_label_count'] = 0
            stats['items_missing_label'] = []
            stats['items_with_multiple_labels'] = []

        elif self.task_type == 'detection':
            bbox_info_template = {
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

            bbox_template = {
                'x': deepcopy(bbox_info_template),
                'y': deepcopy(bbox_info_template),
                'width': deepcopy(bbox_info_template),
                'height': deepcopy(bbox_info_template),
                'area(wxh)': deepcopy(bbox_info_template),
                'ratio(w/h)': deepcopy(bbox_info_template),
                'short': deepcopy(bbox_info_template),
                'long': deepcopy(bbox_info_template)
            }

            stats['total_bbox_count'] = 0
            stats['items_missing_bbox'] = []
            stats['items_with_negative_length'] = {}
            stats['items_with_invalid_value'] = {}
            stats['bbox_distribution_in_label'] = {}
            stats['bbox_distribution_in_attribute'] = {}
            stats['bbox_distribution_in_dataset_item'] = {}

            bbox_dist_by_label = stats['bbox_distribution_in_label']
            bbox_dist_by_attr = stats['bbox_distribution_in_attribute']
            bbox_dist_in_item = stats['bbox_distribution_in_dataset_item']
            items_w_neg_len = stats['items_with_negative_length']
            items_w_invalid_val = stats['items_with_invalid_value']
            _k = self.far_from_mean_thr

            def _update_prop_distributions(ann_bbox_info, target_stats):
                for prop, val in ann_bbox_info.items():
                    prop_stats = target_stats[prop]
                    prop_dist = prop_stats['distribution']
                    prop_stats['distribution'] = np.append(prop_dist, val)

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

            def _update_bbox_stats_by_label(item, ann, bbox_label_stats):
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
                            item.id, {})
                        invalid_props = anns_w_invalid_val.setdefault(
                            ann.id, [])
                        invalid_props.append(prop)

                for prop in ['width', 'height']:
                    val = ann_bbox_info[prop]
                    if val < 1:
                        bbox_has_error = True
                        anns_w_neg_len = items_w_neg_len.setdefault(item.id, {})
                        neg_props = anns_w_neg_len.setdefault(ann.id, {})
                        neg_props[prop] = val

                if not bbox_has_error:
                    _update_prop_distributions(ann_bbox_info, bbox_label_stats)

                return ann_bbox_info, bbox_has_error

            def _compute_prop_stats_from_dist():
                for label_name, bbox_stats in bbox_dist_by_label.items():
                    prop_stats_list = list(bbox_stats.values())
                    bbox_attr_label = bbox_dist_by_attr.get(label_name, {})
                    for attr, vals in bbox_attr_label.items():
                        for val, val_stats in vals.items():
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

            def _is_valid_bbox(item, ann):
                is_bbox = ann.type == self.ann_type
                has_defined_label = 0 <= ann.label < len(label_categories)
                if not is_bbox or not has_defined_label:
                    return False

                bbox_has_neg_len = ann.id in items_w_neg_len.get(item.id, {})
                bbox_has_invalid_val = ann.id in items_w_invalid_val.get(
                    item.id, {})
                return not (bbox_has_neg_len or bbox_has_invalid_val)

            def _far_from_mean(val, mean, stdev):
                return val > mean + (_k * stdev) or val < mean - (_k * stdev)

            def _update_props_far_from_mean(item, ann):
                valid_attrs = label_categories[ann.label].attributes
                label_name = label_categories[ann.label].name
                bbox_label_stats = bbox_dist_by_label[label_name]

                _x, _y, _w, _h = ann.get_bbox()
                area = ann.get_area()
                ratio = _w / _h
                _short = _w if _w < _h else _h
                _long = _w if _w > _h else _h

                ann_bbox_info = _generate_ann_bbox_info(
                    _x, _y, _w, _h, area, ratio, _short, _long)

                for prop, val in ann_bbox_info.items():
                    prop_stats = bbox_label_stats[prop]
                    items_far_from_mean = prop_stats['items_far_from_mean']
                    mean = prop_stats['mean']
                    stdev = prop_stats['stdev']

                    if _far_from_mean(val, mean, stdev):
                        bboxs_far_from_mean = items_far_from_mean.setdefault(
                            item.id, {})
                        bboxs_far_from_mean[ann.id] = val

                for attr, value in ann.attributes.items():
                    if attr in valid_attrs:
                        bbox_attr_stats = bbox_dist_by_attr[label_name][attr]
                        bbox_val_stats = bbox_attr_stats[str(value)]

                        for prop, val in ann_bbox_info.items():
                            prop_stats = bbox_val_stats[prop]
                            items_far_from_mean = \
                                prop_stats['items_far_from_mean']
                            mean = prop_stats['mean']
                            stdev = prop_stats['stdev']

                            if _far_from_mean(val, mean, stdev):
                                bboxs_far_from_mean = \
                                    items_far_from_mean.setdefault(
                                        item.id, {})
                                bboxs_far_from_mean[ann.id] = val

        for category in label_categories:
            defined_label_dist[category.name] = 0

        for item in dataset:
            ann_count = [ann.type == self.ann_type \
                for ann in item.annotations].count(True)

            if self.task_type == 'classification':
                if ann_count == 0:
                    stats['items_missing_label'].append(item.id)
                elif ann_count > 1:
                    stats['items_with_multiple_labels'].append(item.id)
                stats['total_label_count'] += ann_count

            elif self.task_type == 'detection':
                if ann_count < 1:
                    stats['items_missing_bbox'].append(item.id)
                stats['total_bbox_count'] += ann_count
                bbox_dist_in_item[item.id] = ann_count

            for ann in item.annotations:
                if ann.type == self.ann_type:
                    if not 0 <= ann.label < len(label_categories):
                        label_name = ann.label

                        label_stats = undefined_label_dist.setdefault(
                            ann.label, deepcopy(undefined_label_template))
                        label_stats['items_with_undefined_label'].append(
                            item.id)

                        label_stats['count'] += 1
                        valid_attrs = set()
                        missing_attrs = set()
                    else:
                        label_name = label_categories[ann.label].name
                        defined_label_dist[label_name] += 1

                        defined_attr_stats = defined_attr_dist.setdefault(
                            label_name, {})

                        valid_attrs = label_categories[ann.label].attributes
                        ann_attrs = getattr(ann, 'attributes', {}).keys()
                        missing_attrs = valid_attrs.difference(ann_attrs)

                        for attr in valid_attrs:
                            defined_attr_stats.setdefault(
                                attr, deepcopy(defined_attr_template))

                        if self.task_type == 'detection':
                            bbox_label_stats = bbox_dist_by_label.setdefault(
                                label_name, deepcopy(bbox_template))
                            ann_bbox_info, bbox_has_error = \
                                _update_bbox_stats_by_label(
                                    item, ann, bbox_label_stats)

                    for attr in missing_attrs:
                        attr_dets = defined_attr_stats[attr]
                        attr_dets['items_missing_attribute'].append(item.id)

                    for attr, value in ann.attributes.items():
                        if attr not in valid_attrs:
                            undefined_attr_stats = \
                                undefined_attr_dist.setdefault(
                                    label_name, {})
                            attr_dets = undefined_attr_stats.setdefault(
                                attr, deepcopy(undefined_attr_template))
                            attr_dets['items_with_undefined_attr'].append(
                                item.id)
                        else:
                            attr_dets = defined_attr_stats[attr]

                            if self.task_type == 'detection' and \
                                ann.type == self.ann_type:
                                bbox_attr_label = bbox_dist_by_attr.setdefault(
                                    label_name, {})
                                bbox_attr_stats = bbox_attr_label.setdefault(
                                    attr, {})
                                bbox_val_stats = bbox_attr_stats.setdefault(
                                    str(value), deepcopy(bbox_template))

                                if not bbox_has_error:
                                    _update_prop_distributions(
                                        ann_bbox_info, bbox_val_stats)

                        attr_dets['distribution'].setdefault(str(value), 0)
                        attr_dets['distribution'][str(value)] += 1

        if self.task_type == 'detection':
            _compute_prop_stats_from_dist()

            for item in dataset:
                for ann in item.annotations:
                    if _is_valid_bbox(item, ann):
                        _update_props_far_from_mean(item, ann)

        return stats

    def _check_missing_label_categories(self, stats):
        validation_reports = []

        if len(stats['label_distribution']['defined_labels']) == 0:
            validation_reports += self._generate_validation_report(
                MissingLabelCategories, Severity.error)

        return validation_reports

    def _check_missing_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_missing_attr = attr_dets['items_missing_attribute']
        for item_id in items_missing_attr:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details)

        return validation_reports

    def _check_undefined_label(self, label_name, label_stats):
        validation_reports = []

        items_with_undefined_label = label_stats['items_with_undefined_label']
        for item_id in items_with_undefined_label:
            validation_reports += self._generate_validation_report(
                UndefinedLabel, Severity.error, item_id, label_name)

        return validation_reports

    def _check_undefined_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_with_undefined_attr = attr_dets['items_with_undefined_attr']
        for item_id in items_with_undefined_attr:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details)

        return validation_reports

    def _check_label_defined_but_not_found(self, stats):
        validation_reports = []
        count_by_defined_labels = stats['label_distribution']['defined_labels']
        labels_not_found = [label_name \
            for label_name, count in count_by_defined_labels.items() \
                if count == 0]

        for label_name in labels_not_found:
            validation_reports += self._generate_validation_report(
                LabelDefinedButNotFound, Severity.warning, label_name)

        return validation_reports

    def _check_attribute_defined_but_not_found(self, label_name, attr_stats):
        validation_reports = []
        attrs_not_found = [attr_name \
            for attr_name, attr_dets in attr_stats.items() \
                if len(attr_dets['distribution']) == 0]

        for attr_name in attrs_not_found:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                AttributeDefinedButNotFound, Severity.warning, *details)

        return validation_reports

    def _check_only_one_label(self, stats):
        validation_reports = []
        count_by_defined_labels = stats['label_distribution']['defined_labels']
        labels_found = [label_name \
            for label_name, count in count_by_defined_labels.items() \
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

    def _check_few_samples_in_label(self, stats, thr):
        validation_reports = []
        defined_label_dist = stats['label_distribution']['defined_labels']
        labels_with_few_samples = [(label_name, count) \
            for label_name, count in defined_label_dist.items() \
                if 0 < count < thr]

        for label_name, count in labels_with_few_samples:
            validation_reports += self._generate_validation_report(
                FewSamplesInLabel, Severity.warning, label_name, count)

        return validation_reports

    def _check_few_samples_in_attribute(self, label_name,
                                        attr_name, attr_dets, thr):
        validation_reports = []
        attr_values_with_few_samples = [(attr_value, count) \
            for attr_value, count in attr_dets['distribution'].items()  \
                if count < thr]

        for attr_value, count in attr_values_with_few_samples:
            details = (label_name, attr_name, attr_value, count)
            validation_reports += self._generate_validation_report(
                FewSamplesInAttribute, Severity.warning, *details)

        return validation_reports

    def _check_imbalanced_labels(self, stats, thr):
        validation_reports = []

        defined_label_dist = stats['label_distribution']['defined_labels']
        count_by_defined_labels = [count \
            for label, count in defined_label_dist.items()]

        if len(count_by_defined_labels) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_labels)
        count_min = np.min(count_by_defined_labels)
        balance = count_max / count_min if count_min > 0 else float('inf')
        if balance > thr:
            validation_reports += self._generate_validation_report(
                ImbalancedLabels, Severity.warning)

        return validation_reports

    def _check_imbalanced_attribute(self, label_name, attr_name,
                                    attr_dets, thr):
        validation_reports = []

        count_by_defined_attr = list(attr_dets['distribution'].values())
        if len(count_by_defined_attr) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_attr)
        count_min = np.min(count_by_defined_attr)
        balance = count_max / count_min if count_min > 0 else float('inf')
        if balance > thr:
            validation_reports += self._generate_validation_report(
                ImbalancedAttribute, Severity.warning, label_name, attr_name)

        return validation_reports

    def generate_reports(self, stats):
        raise NotImplementedError('Should be implemented in a subclass.')

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]


class ClassificationValidator(_Validator):
    """
    A validator class for classification tasks.

    ...

    Attributes
    ----------
    task_type : str
        'classification'
    ann_type : AnnotationType
        AnnotationType.label
    far_from_mean_thr : float
        None

    Methods
    -------
    compute_statistics(dataset):
        Computes various statistics of the dataset for classification tasks.
    generate_reports(stats):
        Validates the dataset for classification tasks based on its statistics.
    """

    def __init__(self):
        super().__init__('classification', AnnotationType.label)

    def _check_missing_label_annotation(self, stats):
        validation_reports = []

        items_missing_label = stats['items_missing_label']
        for item_id in items_missing_label:
            validation_reports += self._generate_validation_report(
                MissingLabelAnnotation, Severity.warning, item_id)

        return validation_reports

    def _check_multi_label_annotations(self, stats):
        validation_reports = []

        items_with_multiple_labels = stats['items_with_multiple_labels']
        for item_id in items_with_multiple_labels:
            validation_reports += self._generate_validation_report(
                MultiLabelAnnotations, Severity.error, item_id)

        return validation_reports

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
        reports += self._check_missing_label_annotation(stats)
        reports += self._check_multi_label_annotations(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats, 2)
        reports += self._check_imbalanced_labels(stats, 5)

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
                    label_name, attr_name, attr_dets, 2)
                reports += self._check_imbalanced_attribute(
                    label_name, attr_name, attr_dets, 5)
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


class DetectionValidator(_Validator):
    """
    A validator class for detection tasks.

    ...

    Attributes
    ----------
    task_type : str
        'detection'
    ann_type : AnnotationType
        AnnotationType.bbox
    far_from_mean_thr : float
        2.0

    Methods
    -------
    compute_statistics(dataset):
        Computes various statistics of the dataset for detection tasks.
    generate_reports(stats):
        Validates the dataset for detection tasks based on its statistics.
    """

    def __init__(self):
        super().__init__('detection', AnnotationType.bbox, 2.0)

    def _check_imbalanced_bbox_dist_in_label(self, label_name, bbox_label_stats,
                                             thr, topk_ratio):
        validation_reports = []

        for prop, prop_stats in bbox_label_stats.items():
            value_counts = prop_stats['histogram']['counts']
            n_bucket = len(value_counts)
            topk = int(np.around(n_bucket * topk_ratio))

            if topk > 0:
                topk_values = np.sort(value_counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(value_counts)
                if ratio > thr:
                    details = (label_name, prop)
                    validation_reports += self._generate_validation_report(
                        ImbalancedBboxDistInLabel, Severity.warning, *details)

        return validation_reports

    def _check_imbalanced_bbox_dist_in_attr(self, label_name, attr_name,
                                            bbox_attr_stats, thr, topk_ratio):
        validation_reports = []

        for attr_value, value_stats in bbox_attr_stats.items():
            for prop, prop_stats in value_stats.items():
                value_counts = prop_stats['histogram']['counts']
                n_bucket = len(value_counts)
                topk = int(np.around(n_bucket * topk_ratio))

                if topk > 0:
                    topk_values = np.sort(value_counts)[-topk:]
                    ratio = np.sum(topk_values) / np.sum(value_counts)
                    if ratio > thr:
                        details = (label_name, attr_name, attr_value, prop)
                        validation_reports += self._generate_validation_report(
                            ImbalancedBboxDistInAttribute, 
                            Severity.warning,
                            *details
                        )

        return validation_reports

    def _check_missing_bbox_annotation(self, stats):
        validation_reports = []

        items_missing_bbox = stats['items_missing_bbox']
        for item_id in items_missing_bbox:
            validation_reports += self._generate_validation_report(
                MissingBboxAnnotation, Severity.warning, item_id)

        return validation_reports

    def _check_negative_length(self, stats):
        validation_reports = []

        items_w_neg_len = stats['items_with_negative_length']
        for item_id, anns_w_neg_len in items_w_neg_len.items():
            for ann_id, props in anns_w_neg_len.items():
                for prop, val in props.items():
                    val = round(val, 2)
                    details = (ann_id, prop, val)
                    validation_reports += self._generate_validation_report(
                        NegativeLength, Severity.error, item_id, *details)

        return validation_reports

    def _check_invalid_value(self, stats):
        validation_reports = []

        items_w_invalid_val = stats['items_with_invalid_value']
        for item_id, anns_w_invalid_val in items_w_invalid_val.items():
            for ann_id, props in anns_w_invalid_val.items():
                for prop in props:
                    details = (ann_id, prop)
                    validation_reports += self._generate_validation_report(
                        InvalidValue, Severity.error, item_id, *details)

        return validation_reports

    def _check_far_from_label_mean(self, label_name, bbox_label_stats):
        validation_reports = []

        for prop, prop_stats in bbox_label_stats.items():
            items_far_from_mean = prop_stats['items_far_from_mean']
            if prop_stats['mean'] is not None:
                mean = round(prop_stats['mean'], 2)

            for item_id, anns_far_from_mean in items_far_from_mean.items():
                for ann_id, val in anns_far_from_mean.items():
                    val = round(val, 2)
                    details = (label_name, ann_id, prop, mean, val)
                    validation_reports += self._generate_validation_report(
                        FarFromLabelMean, Severity.warning, item_id, *details)

        return validation_reports

    def _check_far_from_attr_mean(self, label_name, attr_name, bbox_attr_stats):
        validation_reports = []

        for attr_value, value_stats in bbox_attr_stats.items():
            for prop, prop_stats in value_stats.items():
                items_far_from_mean = prop_stats['items_far_from_mean']
                if prop_stats['mean'] is not None:
                    mean = round(prop_stats['mean'], 2)

                for item_id, anns_far_from_mean in items_far_from_mean.items():
                    for ann_id, val in anns_far_from_mean.items():
                        val = round(val, 2)
                        details = (label_name, ann_id, attr_name, attr_value,
                            prop, mean, val)
                        validation_reports += self._generate_validation_report(
                            FarFromAttrMean,
                            Severity.warning,
                            item_id,
                            *details
                        )

        return validation_reports

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
        reports += self._check_missing_bbox_annotation(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats, 2)
        reports += self._check_imbalanced_labels(stats, 5)
        reports += self._check_negative_length(stats)
        reports += self._check_invalid_value(stats)

        label_dist = stats['label_distribution']
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        undefined_label_dist = label_dist['undefined_labels']
        undefined_attr_dist = attr_dist['undefined_attributes']

        bbox_dist_by_label = stats['bbox_distribution_in_label']
        bbox_dist_by_attr = stats['bbox_distribution_in_attribute']

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(
                label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_few_samples_in_attribute(
                    label_name, attr_name, attr_dets, 2)
                reports += self._check_imbalanced_attribute(
                    label_name, attr_name, attr_dets, 5)
                reports += self._check_only_one_attribute_value(
                    label_name, attr_name, attr_dets)
                reports += self._check_missing_attribute(
                    label_name, attr_name, attr_dets)

            bbox_label_stats = bbox_dist_by_label[label_name]
            bbox_attr_label = bbox_dist_by_attr.get(label_name, {})

            reports += self._check_far_from_label_mean(
                label_name, bbox_label_stats)
            reports += self._check_imbalanced_bbox_dist_in_label(
                label_name, bbox_label_stats, 1, 0.25)

            for attr_name, bbox_attr_stats in bbox_attr_label.items():
                reports += self._check_far_from_attr_mean(
                    label_name, attr_name, bbox_attr_stats)
                reports += self._check_imbalanced_bbox_dist_in_attr(
                    label_name, attr_name, bbox_attr_stats, 1, 0.25)

        for label_name, label_stats in undefined_label_dist.items():
            reports += self._check_undefined_label(label_name, label_stats)

        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(
                    label_name, attr_name, attr_dets)

        return reports


def validate_annotations(dataset, task_type):
    """
    Returns the validation results of a dataset based on task type.

    Args:
        dataset (IDataset): Dataset to be validated
        task_type (str): Type of task (ie. 'classification', 'detection', etc.)

    Raises:
        ValueError: 'Invalid task type.'
        ValueError: 'Invalid Dataset type.'

    Returns:
        validation_results (dict):
            Dict with validation statistics, reports and summary.

    """

    validation_results = {}

    if task_type == 'classification':
        validator = ClassificationValidator()
    elif task_type == 'detection':
        validator = DetectionValidator()
    else:
        raise ValueError('Invalid task type.')

    if not isinstance(dataset, IDataset):
        raise ValueError('Invalid Dataset type.')

    # generate statistics
    stats = validator.compute_statistics(dataset)
    validation_results['statistics'] = stats

    # generate validation reports and summary
    reports = validator.generate_reports(stats)
    reports = list(map(lambda r : r.to_dict(), reports))

    summary = {
        'errors': sum(map(lambda r : r['severity'] == 'error', reports)),
        'warnings': sum(map(lambda r : r['severity'] == 'warning', reports))
    }

    validation_results['validation_reports'] = reports
    validation_results['summary'] = summary

    return validation_results
