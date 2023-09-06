# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import logging as log
from typing import List, Set

import numpy as np

from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetItem
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
    DatasetValidationError,
    FarFromAttrMean,
    FarFromLabelMean,
    FewSamplesInAttribute,
    FewSamplesInLabel,
    ImbalancedAttribute,
    ImbalancedDistInAttribute,
    ImbalancedDistInLabel,
    ImbalancedLabels,
    InvalidValue,
    LabelDefinedButNotFound,
    MissingAnnotation,
    MissingAttribute,
    MissingLabelCategories,
    MultiLabelAnnotations,
    NegativeLength,
    OnlyOneAttributeValue,
    OnlyOneLabel,
    UndefinedAttribute,
    UndefinedLabel,
)
from datumaro.components.validator import Severity, TaskType, Validator


class _BaseAnnStats:
    ATTR_WARNINGS = {
        AttributeDefinedButNotFound,
        FewSamplesInAttribute,
        ImbalancedAttribute,
        MissingAttribute,
        OnlyOneAttributeValue,
        UndefinedAttribute,
    }

    def __init__(self, label_categories: LabelCategories, warnings: List[DatasetValidationError]):
        self.label_categories = label_categories
        self.warnings = set(warnings)

        self.stats = {}
        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats.setdefault(label_name, {"cnt": 0, "attributes": {}})
            self.stats[label_name]["attributes"] = {attr: {} for attr in label_cat.attributes}

        self.stats["undefined_attribute"] = set()  # (item_key, label_name, attr_name)
        self.stats["undefined_label"] = set()  # (item_key, label_name)
        self.stats["missing_attribute"] = set()  # (item_key, label_name, attr_name)
        self.stats["missing_label"] = set()  # (item_key)

    def update_item(self, item: DatasetItem):
        item_key = (item.id, item.subset)

        if {MissingAnnotation} & self.warnings:
            if sum([1 for ann in item.annotations]) == 0:
                self.stats["missing_label"].add(item_key)

        self._update_item_type_stats(item)

    def update_ann(self, item_key: tuple, annotation: Annotation):
        if annotation.label in self.label_categories:
            self._update_ann_type_stats(item_key, annotation)

            label_name = self.label_categories[annotation.label].name
            self.stats[label_name]["cnt"] += 1

            if self.ATTR_WARNINGS & self.warnings:
                for attr, value in annotation.attributes.items():
                    try:
                        # attribute is defined according to the label
                        if self.stats[label_name]["attributes"][attr].get(str(value), None):
                            self.stats[label_name]["attributes"][attr][str(value)] += 1
                        else:
                            self.stats[label_name]["attributes"][attr][str(value)] = 1
                    except Exception as e:
                        log.warning(
                            "Label '%s': failed to get access to attribute %e" % (label_name, e)
                        )
                        if {UndefinedAttribute} & self.warnings:
                            # attribute is not defined within the label
                            self.stats["undefined_attribute"].add(item_key + (label_name, attr))

                if {MissingAttribute} & self.warnings:
                    # defined attribute is not found in this annotation
                    for attr in self.stats[label_name]["attributes"]:
                        if attr not in annotation.attributes:
                            self.stats["missing_attribute"].add(item_key + (label_name, attr))
        else:
            label_name = str(annotation.label)
            if {UndefinedLabel} & self.warnings:
                self.stats["undefined_label"].add(item_key + (label_name,))

            if {UndefinedAttribute} & self.warnings:
                for attr, value in annotation.attributes.items():
                    self.stats["undefined_attribute"].add(item_key + (label_name, attr))

    def _update_item_type_stats(self, item: DatasetItem):
        NotImplemented

    def _update_ann_type_stats(self, item_key: tuple, annotation: Annotation):
        NotImplemented


class ClsStats(_BaseAnnStats):
    def __init__(self, label_categories: LabelCategories, warnings: set):
        super().__init__(label_categories=label_categories, warnings=warnings)

        self.stats["multiple_label"] = set()

    def _update_item_type_stats(self, item: DatasetItem):
        item_key = (item.id, item.subset)

        if {MultiLabelAnnotations} & self.warnings:
            if sum([1 for ann in item.annotations if ann.type == AnnotationType.label]) > 1:
                self.stats["multiple_label"].add(item_key)

    def _update_ann_type_stats(self, item_key: tuple, annotation: Annotation):
        pass


class DetStats(_BaseAnnStats):
    BBOX_WARNINGS = {
        FarFromAttrMean,  # annotation level
        FarFromLabelMean,  # annotation level
        ImbalancedDistInAttribute,  # annotation level: bbox
        ImbalancedDistInLabel,  # annotation level: bbox
        InvalidValue,  # annotation level
        NegativeLength,  # annotation level
    }

    def __init__(self, label_categories: LabelCategories, warnings: set):
        super().__init__(label_categories=label_categories, warnings=warnings)

        self.items = {}
        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats[label_name]["pts"] = []  # (id, subset, ann_id, w, h)

        self.stats["invalid_value"] = set()
        self.stats["negative_length"] = set()

    def _update_item_type_stats(self, item: DatasetItem):
        pass

    def _update_ann_type_stats(self, item_key: tuple, annotation: Annotation):
        if self.BBOX_WARNINGS & self.warnings:
            _, _, w, h = annotation.get_bbox()

            if {InvalidValue} & self.warnings:
                if w == float("inf") or np.isnan(w) or h == float("inf") or np.isnan(h):
                    self.stats["invalid_value"].add(item_key + (annotation.id,))

            if {NegativeLength} & self.warnings:
                if w < 1 or h < 1:
                    self.stats["negative_length"].add(item_key + (annotation.id,))

            label_name = self.label_categories[annotation.label].name
            self.stats[label_name]["pts"].append(item_key + (annotation.id, w, h))


class SegStats(_BaseAnnStats):
    SEG_WARNINGS = {
        FarFromAttrMean,  # annotation level
        FarFromLabelMean,  # annotation level
        ImbalancedDistInAttribute,  # annotation level: bbox
        ImbalancedDistInLabel,  # annotation level: bbox
    }

    def __init__(self, label_categories: LabelCategories, warnings: set):
        super().__init__(label_categories=label_categories, warnings=warnings)

        for label_cat in self.label_categories.items:
            label_name = label_cat.name
            self.stats[label_name]["pts"] = []

    def _update_item_type_stats(self, item: DatasetItem):
        pass

    def _update_ann_type_stats(self, item_key: tuple, annotation: Annotation):
        if self.SEG_WARNINGS & self.warnings:
            _, _, w, h = annotation.get_bbox()

            label_name = self.label_categories[annotation.label].name
            self.stats[label_name]["pts"].append(item_key + (annotation.id, w, h))


class ConfigurableValidator(Validator, CliPlugin):
    DEFAULT_FEW_SAMPLES_THR = 1
    DEFAULT_IMBALANCE_RATIO_THR = 50
    DEFAULT_FAR_FROM_MEAN_THR = 5
    DEFAULT_DOMINANCE_RATIO_THR = 0.8
    DEFAULT_TOPK_BINS = 0.1

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-fs",
            "--few-samples-thr",
            default=cls.DEFAULT_FEW_SAMPLES_THR,
            type=int,
            help="Threshold for giving a warning for minimum number of "
            "samples per class (default: %(default)s)",
        )
        parser.add_argument(
            "-ir",
            "--imbalance-ratio-thr",
            default=cls.DEFAULT_IMBALANCE_RATIO_THR,
            type=int,
            help="Threshold for giving data imbalance warning. "
            "IR(imbalance ratio) = majority/minority "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-m",
            "--far-from-mean-thr",
            default=cls.DEFAULT_FAR_FROM_MEAN_THR,
            type=float,
            help="Threshold for giving a warning that data is far from mean. "
            "A constant used to define mean +/- k * standard deviation "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-dr",
            "--dominance-ratio-thr",
            default=cls.DEFAULT_DOMINANCE_RATIO_THR,
            type=float,
            help="Threshold for giving a warning for bounding box imbalance. "
            "Dominace_ratio = ratio of Top-k bin to total in histogram "
            "(default: %(default)s)",
        )
        parser.add_argument(
            "-k",
            "--topk-bins",
            default=cls.DEFAULT_TOPK_BINS,
            type=float,
            help="Ratio of bins with the highest number of data "
            "to total bins in the histogram. A value in the range [0, 1] "
            "(default: %(default)s)",
        )
        return parser

    def __init__(
        self,
        tasks: List[TaskType],
        warnings: Set[DatasetValidationError],
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
        self.tasks = tasks
        self.warnings = warnings

        self.init_flag = False
        self.all_stats = {task: None for task in tasks}

        self.few_samples_thr = few_samples_thr if few_samples_thr else self.DEFAULT_FEW_SAMPLES_THR
        self.imbalance_ratio_thr = (
            imbalance_ratio_thr if imbalance_ratio_thr else self.DEFAULT_IMBALANCE_RATIO_THR
        )
        self.far_from_mean_thr = (
            far_from_mean_thr if far_from_mean_thr else self.DEFAULT_FAR_FROM_MEAN_THR
        )
        self.dominance_thr = (
            dominance_ratio_thr if dominance_ratio_thr else self.DEFAULT_DOMINANCE_RATIO_THR
        )
        self.topk_bins_ratio = topk_bins if topk_bins else self.DEFAULT_TOPK_BINS

    def _init_stats_collector(self, label_categories):
        for task in self.tasks:
            if task == TaskType.classification:
                self.all_stats[task] = ClsStats(
                    label_categories=label_categories, warnings=self.warnings
                )
            elif task == TaskType.detection:
                self.all_stats[task] = DetStats(
                    label_categories=label_categories, warnings=self.warnings
                )
            elif task == TaskType.segmentation:
                self.all_stats[task] = SegStats(
                    label_categories=label_categories, warnings=self.warnings
                )
        self.init_flag = True

    def _get_stats_collector(self, ann_type):
        if TaskType.classification in self.tasks and ann_type == AnnotationType.label:
            return self.all_stats[TaskType.classification]
        elif TaskType.detection in self.tasks and ann_type == AnnotationType.bbox:
            return self.all_stats[TaskType.detection]
        elif TaskType.segmentation in self.tasks and ann_type in [
            AnnotationType.mask,
            AnnotationType.polygon,
            AnnotationType.ellipse,
        ]:
            return self.all_stats[TaskType.segmentation]
        else:
            return None

    def compute_statistics(self, dataset):
        if not self.init_flag:
            self._label_categories = dataset.categories()[AnnotationType.label]
            self._init_stats_collector(label_categories=self._label_categories)

        for item in dataset:
            for stats_collector in self.all_stats.values():
                if not stats_collector:
                    continue
                stats_collector.update_item(item)

            item_key = (item.id, item.subset)
            for annotation in item.annotations:
                stats_collector = self._get_stats_collector(ann_type=annotation.type)
                if not stats_collector:
                    continue
                stats_collector.update_ann(item_key, annotation)

        return {task: stats_collector.stats for task, stats_collector in self.all_stats.items()}

    def generate_reports(self, task_stats):
        reports = {}

        for task, stats in task_stats.items():
            reports[task] = []
            if task == TaskType.classification:
                if {MultiLabelAnnotations} & self.warnings:
                    reports[task] += self._check_multiple_label(stats)

            elif task == TaskType.detection:
                if {InvalidValue} & self.warnings:
                    reports[task] += self._check_invalid_value(stats)
                if {NegativeLength} & self.warnings:
                    reports[task] += self._check_negative_length(stats)
                if {FarFromLabelMean} & self.warnings:
                    reports[task] += self._check_far_from_mean(stats)
                if {ImbalancedDistInLabel} & self.warnings:
                    reports[task] += self._check_imbalanced_dist_in_label(stats)
                # if {ImbalancedDistInLabel} & self.warnings:
                #     reports[task] += self._check_negative_length(stats)
                # if {ImbalancedDistInAttribute} & self.warnings:
                #     reports[task] += self._check_negative_length(stats)

            elif task == TaskType.segmentation:
                if {FarFromLabelMean} & self.warnings:
                    reports[task] += self._check_far_from_mean(stats)
                if {ImbalancedDistInLabel} & self.warnings:
                    reports[task] += self._check_imbalanced_dist_in_label(stats)

            # report for dataset
            if {MissingLabelCategories} & self.warnings:
                reports[task] += self._check_missing_label_categories(stats)

            # report for item
            if {MissingAnnotation} & self.warnings:
                reports[task] += self._check_missing_label(stats)

            # report for label
            if {UndefinedLabel} & self.warnings:
                reports[task] += self._check_undefined_label(stats)
            if {LabelDefinedButNotFound} & self.warnings:
                reports[task] += self._check_label_defined_but_not_found(stats)
            if {OnlyOneLabel} & self.warnings:
                reports[task] += self._check_only_one_label(stats)
            if {FewSamplesInLabel} & self.warnings:
                reports[task] += self._check_few_samples_in_label(stats)
            if {ImbalancedLabels} & self.warnings:
                reports[task] += self._check_imbalanced_labels(stats)

            # report for attributes
            if {UndefinedAttribute} & self.warnings:
                reports[task] += self._check_undefined_attribute(stats)
            if {AttributeDefinedButNotFound} & self.warnings:
                reports[task] += self._check_attribute_defined_but_not_found(stats)
            if {OnlyOneAttributeValue} & self.warnings:
                reports[task] += self._check_only_one_attribute(stats)
            if {MissingAttribute} & self.warnings:
                reports[task] += self._check_missing_attribute(stats)
            if {FewSamplesInAttribute} & self.warnings:
                reports[task] += self._check_few_samples_in_attribute(stats)
            if {ImbalancedAttribute} & self.warnings:
                reports[task] += self._check_imbalanced_attribute(stats)

        return reports

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]

    def _check_missing_label_categories(self, stats):
        validation_reports = []

        if len(self._label_categories.items) == 0:
            validation_reports += self._generate_validation_report(
                MissingLabelCategories, Severity.error
            )

        return validation_reports

    def _check_multiple_label(self, stats):
        validation_reports = []

        items_multiple_label = stats["multiple_label"]
        for item_id, item_subset in items_multiple_label:
            validation_reports += self._generate_validation_report(
                MultiLabelAnnotations, Severity.error, item_id, item_subset
            )

        return validation_reports

    def _check_missing_label(self, stats):
        validation_reports = []

        items_missing_label = stats["missing_label"]
        for item_id, item_subset in items_missing_label:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset, "label"
            )

        return validation_reports

    def _check_missing_attribute(self, stats):
        validation_reports = []

        items_missing_attr = stats["missing_attribute"]
        for item_id, item_subset, label_name, attr_name in items_missing_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details
            )

        return validation_reports

    def _check_undefined_label(self, stats):
        validation_reports = []

        items_undefined_label = stats["undefined_label"]
        for item_id, item_subset, label_name in items_undefined_label:
            details = (item_subset, label_name)
            validation_reports += self._generate_validation_report(
                UndefinedLabel, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_undefined_attribute(self, stats):
        validation_reports = []

        items_undefined_attr = stats["undefined_attribute"]
        for item_id, item_subset, label_name, attr_name in items_undefined_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_label_defined_but_not_found(self, stats):
        validation_reports = []

        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            if stats[label_name]["cnt"] == 0:
                validation_reports += self._generate_validation_report(
                    LabelDefinedButNotFound, Severity.warning, label_name
                )

        return validation_reports

    def _check_attribute_defined_but_not_found(self, stats):
        validation_reports = []

        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            for attr_name, dist in stats[label_name]["attributes"].items():
                if dist:
                    continue
                details = (label_name, attr_name)
                validation_reports += self._generate_validation_report(
                    AttributeDefinedButNotFound, Severity.warning, *details
                )

        return validation_reports

    def _check_only_one_label(self, stats):
        labels_found = []
        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            if stats[label_name]["cnt"] > 0:
                labels_found.append(label_name)

        validation_reports = []
        if len(labels_found) == 1:
            validation_reports += self._generate_validation_report(
                OnlyOneLabel, Severity.info, labels_found[0]
            )

        return validation_reports

    def _check_only_one_attribute(self, stats):
        validation_reports = []
        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            for attr_name, values in stats[label_name]["attributes"].items():
                for attr_value, cnt in values.items():
                    if cnt != 1:
                        continue
                    details = (label_name, attr_name, attr_value)
                    validation_reports += self._generate_validation_report(
                        OnlyOneAttributeValue, Severity.info, *details
                    )

        return validation_reports

    def _check_few_samples_in_label(self, stats):
        validation_reports = []
        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            if stats[label_name]["cnt"] < self.few_samples_thr:
                validation_reports += self._generate_validation_report(
                    FewSamplesInLabel, Severity.info, label_name, stats[label_name]["cnt"]
                )

        return validation_reports

    def _check_few_samples_in_attribute(self, stats):
        validation_reports = []
        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            for attr_name, values in stats[label_name]["attributes"].items():
                for attr_value, cnt in values.items():
                    if cnt >= self.few_samples_thr:
                        continue
                    details = (label_name, attr_name, attr_value, cnt)
                    validation_reports += self._generate_validation_report(
                        FewSamplesInAttribute, Severity.info, *details
                    )

        return validation_reports

    def _check_imbalanced_labels(self, stats):
        validation_reports = []

        counts = [stats[label_cat.name]["cnt"] for label_cat in self._label_categories.items]

        if len(counts) == 0:
            return validation_reports

        count_max = np.max(counts)
        count_min = np.min(counts)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= self.imbalance_ratio_thr:
            validation_reports += self._generate_validation_report(ImbalancedLabels, Severity.info)

        return validation_reports

    def _check_imbalanced_attribute(self, stats):
        validation_reports = []

        for label_cat in self._label_categories.items:
            label_name = label_cat.name
            for attr_name, attr_vals in stats[label_name]["attributes"].items():
                counts = [cnt for cnt in attr_vals.values()]

                if len(counts) == 0:
                    continue

                count_max = np.max(counts)
                count_min = np.min(counts)
                balance = count_max / count_min if count_min > 0 else float("inf")
                if balance >= self.imbalance_ratio_thr:
                    validation_reports += self._generate_validation_report(
                        ImbalancedAttribute, Severity.info, label_name, attr_name
                    )

        return validation_reports

    def _check_negative_length(self, stats):
        validation_reports = []

        items_neg_len = stats["negative_length"]
        for item_id, item_subset, ann_id in items_neg_len:
            details = (item_subset, ann_id)
            validation_reports += self._generate_validation_report(
                NegativeLength, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_invalid_value(self, stats):
        validation_reports = []

        items_invalid_val = stats["invalid_value"]
        for item_id, item_subset, ann_id in items_invalid_val:
            details = (item_subset, ann_id)
            validation_reports += self._generate_validation_report(
                InvalidValue, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_far_from_mean(self, stats):
        def _far_from_mean(val, mean, stdev):
            thr = self.far_from_mean_thr
            return val > mean + (thr * stdev) or val < mean - (thr * stdev)

        validation_reports = []

        for label_cat in self._label_categories.items:
            label_name = label_cat.name

            prop = {"width": [], "height": [], "ratio": [], "area": []}
            for _, _, _, w, h in stats[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["ratio"].append(w / h)
                prop["area"].append(w * h)

            prop_stats = {}
            for p, vals in prop.items():
                prop_stats[p] = {}
                prop_stats[p]["mean"] = np.mean(vals)
                prop_stats[p]["stdev"] = np.std(vals)

            for item_id, item_subset, ann_id, w, h in stats[label_name]["pts"]:
                item_prop = {"width": w, "height": h, "ratio": w / h, "area": w * h}

                for p in prop_stats.keys():
                    if _far_from_mean(item_prop[p], prop_stats[p]["mean"], prop_stats[p]["stdev"]):
                        details = (
                            item_subset,
                            label_name,
                            ann_id,
                            f"bbox {p}",
                            prop_stats[p]["mean"],
                            item_prop[p],
                        )
                        validation_reports += self._generate_validation_report(
                            FarFromLabelMean, Severity.warning, item_id, *details
                        )

        return validation_reports

    def _check_imbalanced_dist_in_label(self, stats):
        validation_reports = []

        for label_cat in self._label_categories.items:
            label_name = label_cat.name

            prop = {"width": [], "height": [], "ratio": [], "area": []}
            for _, _, _, w, h in stats[label_name]["pts"]:
                prop["width"].append(w)
                prop["height"].append(h)
                prop["ratio"].append(w / h)
                prop["area"].append(w * h)

            for p, vals in prop.items():
                counts, _ = np.histogram(vals)

                n_bucket = len(counts)
                if n_bucket < 2:
                    continue

                topk = max(1, int(np.around(n_bucket * self.topk_bins_ratio)))

                topk_values = np.sort(counts)[-topk:]
                ratio = np.sum(topk_values) / np.sum(counts)
                if ratio >= self.dominance_thr:
                    details = (label_name, f"bbox {p}")
                    validation_reports += self._generate_validation_report(
                        ImbalancedDistInLabel, Severity.info, *details
                    )

        return validation_reports
