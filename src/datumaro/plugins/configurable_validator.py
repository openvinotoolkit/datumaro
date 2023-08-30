# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from copy import deepcopy

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import (
    AttributeDefinedButNotFound,
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

DEFAULT_LABEL_GROUP = "default"


class ValidationStatsCallback:
    def __init__(self, annotation_types, warnings):
        self.stats = {}
        if [
            MissingLabelCategories,
            LabelDefinedButNotFound,
            OnlyOneLabel,
            FewSamplesInLabel,
            ImbalancedLabels,
            UndefinedLabel,
            FarFromLabelMean,
            ImbalancedDistInLabel,
        ] in warnings:
            self.stats["label"] = {
                "defined": defaultdict(lambda: 0),
                "undefined": [],
                "point_dist": {},
            }

        if [
            AttributeDefinedButNotFound,
            MissingAttribute,
            OnlyOneAttributeValue,
            FewSamplesInAttribute,
            ImbalancedAttribute,
            UndefinedAttribute,
            FarFromAttrMean,
            ImbalancedDistInAttribute,
        ] in warnings:
            self.stats["attribute"] = {
                "defined": defaultdict(lambda: 0),
                "undefined": [],
                "point_dist": {},
            }

        if [MissingAnnotation, MultiLabelAnnotations, NegativeLength, InvalidValue] in warnings:
            self.stats["item"] = {
                "missing_annotation": [],
                "multiple_labels": [],
                "negative_length": [],
                "invalid_value": [],
            }

        self.annotation_type = annotation_types

    def init_label_categories(self, label_categories):
        self.label_categories = label_categories

    def _is_desirable_type(self, ann_type: AnnotationType) -> bool:
        if ann_type in self.annotation_type:
            return True

    def _is_desirable_validation(self, val_type: str) -> bool:
        if val_type in self.stats.keys():
            return True

    def update_stats(self, item):
        item_key = (item.id, item.subset)

        if self._is_desirable_type("item"):
            ann_cnt = sum(1 for ann in item.annotations if self._is_desirable_type(ann.type))
            if ann_cnt == 0:
                self.stats["item"]["missing_annotation"].append(item_key)

            ann_cnt = sum(1 for ann in item.annotations if ann.type == AnnotationType.label)
            if ann_cnt > 1:
                self.stats["item"]["multiple_labels"].append(item.key)

        for ann in item.annotations:
            if not self._is_desirable_type(ann.type):
                continue

            if self._is_desirable_type("label"):
                if ann.label not in self.label_categories:
                    self.stats["label"]["undefined"].append(item_key)
                    valid_attrs = set()
                    missing_attrs = set()
                else:
                    label_name = self.label_categories[ann.label].name
                    self.stats["label"]["defined"][label_name] += 1

                    defined_attr_stats = self.stats["attribute"]["defined"].setdefault(
                        label_name, {}
                    )

                    label_attrs = self.label_categories[ann.label].attributes
                    valid_attrs = self.label_categories.attributes.union(label_attrs)
                    ann_attrs = getattr(ann, "attributes", {}).keys()
                    missing_attrs = valid_attrs.difference(ann_attrs)

                    defined_attr_template = {"items_missing_attribute": [], "distribution": {}}
                    for attr in valid_attrs:
                        defined_attr_stats.setdefault(attr, deepcopy(defined_attr_template))

            if self._is_desirable_type("attribute"):
                for attr in missing_attrs:
                    attr_dets = defined_attr_stats[attr]
                    attr_dets["items_missing_attribute"].append(item_key)

                undefined_attr_template = {"items_with_undefined_attr": [], "distribution": {}}
                for attr, value in ann.attributes.items():
                    if attr not in valid_attrs:
                        undefined_attr_stats = self.stats["attribute"]["undefined"].setdefault(
                            label_name, {}
                        )
                        attr_dets = undefined_attr_stats.setdefault(
                            attr, deepcopy(undefined_attr_template)
                        )
                        attr_dets["items_with_undefined_attr"].append(item_key)
                    else:
                        attr_dets = defined_attr_stats[attr]

                    attr_dets["distribution"].setdefault(str(value), 0)
                    attr_dets["distribution"][str(value)] += 1


class ConfigurableValidator(Validator, CliPlugin):
    DEFAULT_FEW_SAMPLES_THR = 1
    DEFAULT_IMBALANCE_RATIO_THR = 50
    DEFAULT_FAR_FROM_MEAN_THR = 5
    DEFAULT_DOMINANCE_RATIO_THR = 0.8
    DEFAULT_TOPK_BINS = 0.1

    # statistics templates
    numerical_stat_template = {
        "items_far_from_mean": {},
        "mean": None,
        "stdev": None,
        "min": None,
        "max": None,
        "median": None,
        "histogram": {
            "bins": [],
            "counts": [],
        },
        "distribution": [],
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
        parser.add_argument(
            "-ann",
            "--annotation-type",
            default=[AnnotationType.label],
            type=list,
            help="Threshold for giving a warning for minimum number of "
            "samples per class (default: %(default)s)",
        )
        parser.add_argument(
            "-val",
            "--validation-type",
            default=cls.DEFAULT_FEW_SAMPLES_THR,
            type=list,
            help="Threshold for giving a warning for minimum number of "
            "samples per class (default: %(default)s)",
        )
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
        task_type: list,
        validation_report: list,
        few_samples_thr=None,
        imbalance_ratio_thr=None,
        far_from_mean_thr=None,
        dominance_ratio_thr=None,
        topk_bins=None,
    ):
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

        if task_type == TaskType.classification:
            self.ann_types = {AnnotationType.label}
            self.str_ann_type = "label"
        elif task_type == TaskType.detection:
            self.ann_types = {AnnotationType.bbox}
            self.str_ann_type = "bounding box"
        elif task_type == TaskType.segmentation:
            self.ann_types = {AnnotationType.mask, AnnotationType.polygon, AnnotationType.ellipse}
            self.str_ann_type = "mask or polygon or ellipse"

        if few_samples_thr is None:
            few_samples_thr = self.DEFAULT_FEW_SAMPLES_THR

        if imbalance_ratio_thr is None:
            imbalance_ratio_thr = self.DEFAULT_IMBALANCE_RATIO_THR

        if far_from_mean_thr is None:
            far_from_mean_thr = self.DEFAULT_FAR_FROM_MEAN_THR

        if dominance_ratio_thr is None:
            dominance_ratio_thr = self.DEFAULT_DOMINANCE_RATIO_THR

        if topk_bins is None:
            topk_bins = self.DEFAULT_TOPK_BINS

        self.few_samples_thr = few_samples_thr
        self.imbalance_ratio_thr = imbalance_ratio_thr
        self.far_from_mean_thr = far_from_mean_thr
        self.dominance_thr = dominance_ratio_thr
        self.topk_bins_ratio = topk_bins

    def _compute_common_statistics(self, dataset):
        defined_attr_template = {"items_missing_attribute": [], "distribution": {}}
        undefined_attr_template = {"items_with_undefined_attr": [], "distribution": {}}
        undefined_label_template = {
            "count": 0,
            "items_with_undefined_label": [],
        }

        stats = {
            "label_distribution": {
                "defined_labels": {},
                "undefined_labels": {},
            },
            "attribute_distribution": {"defined_attributes": {}, "undefined_attributes": {}},
        }
        stats["total_ann_count"] = 0
        stats["items_missing_annotation"] = []

        label_dist = stats["label_distribution"]
        defined_label_dist = label_dist["defined_labels"]
        undefined_label_dist = label_dist["undefined_labels"]

        attr_dist = stats["attribute_distribution"]
        defined_attr_dist = attr_dist["defined_attributes"]
        undefined_attr_dist = attr_dist["undefined_attributes"]

        label_categories = dataset.categories().get(AnnotationType.label, LabelCategories())
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
                stats["items_missing_annotation"].append(item_key)
            stats["total_ann_count"] += ann_count

            for ann in annotations:
                if not 0 <= ann.label < len(label_categories):
                    label_name = ann.label

                    label_stats = undefined_label_dist.setdefault(
                        ann.label, deepcopy(undefined_label_template)
                    )
                    label_stats["items_with_undefined_label"].append(item_key)

                    label_stats["count"] += 1
                    valid_attrs = set()
                    missing_attrs = set()
                else:
                    label_name = label_categories[ann.label].name
                    defined_label_dist[label_name] += 1

                    defined_attr_stats = defined_attr_dist.setdefault(label_name, {})

                    valid_attrs = base_valid_attrs.union(label_categories[ann.label].attributes)
                    ann_attrs = getattr(ann, "attributes", {}).keys()
                    missing_attrs = valid_attrs.difference(ann_attrs)

                    for attr in valid_attrs:
                        defined_attr_stats.setdefault(attr, deepcopy(defined_attr_template))

                for attr in missing_attrs:
                    attr_dets = defined_attr_stats[attr]
                    attr_dets["items_missing_attribute"].append(item_key)

                for attr, value in ann.attributes.items():
                    if attr not in valid_attrs:
                        undefined_attr_stats = undefined_attr_dist.setdefault(label_name, {})
                        attr_dets = undefined_attr_stats.setdefault(
                            attr, deepcopy(undefined_attr_template)
                        )
                        attr_dets["items_with_undefined_attr"].append(item_key)
                    else:
                        attr_dets = defined_attr_stats[attr]

                    attr_dets["distribution"].setdefault(str(value), 0)
                    attr_dets["distribution"][str(value)] += 1

        return stats, filtered_anns

    def _generate_common_reports(self, stats):
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

        # report for dataset
        reports += self._check_missing_label_categories(stats)

        # report for item
        reports += self._check_missing_annotation(stats)

        # report for label
        reports += self._check_undefined_label(stats)
        reports += self._check_label_defined_but_not_found(stats)
        reports += self._check_only_one_label(stats)
        reports += self._check_few_samples_in_label(stats)
        reports += self._check_imbalanced_labels(stats)

        # report for attributes
        attr_dist = stats["attribute_distribution"]
        defined_attr_dist = attr_dist["defined_attributes"]

        defined_labels = defined_attr_dist.keys()
        for label_name in defined_labels:
            attr_stats = defined_attr_dist[label_name]

            reports += self._check_attribute_defined_but_not_found(label_name, attr_stats)

            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_missing_attribute(label_name, attr_name, attr_dets)
                reports += self._check_only_one_attribute(label_name, attr_name, attr_dets)
                reports += self._check_few_samples_in_attribute(label_name, attr_name, attr_dets)
                reports += self._check_imbalanced_attribute(label_name, attr_name, attr_dets)

        undefined_attr_dist = attr_dist["undefined_attributes"]
        for label_name, attr_stats in undefined_attr_dist.items():
            for attr_name, attr_dets in attr_stats.items():
                reports += self._check_undefined_attribute(label_name, attr_name, attr_dets)

        return reports

    def _generate_validation_report(self, error, *args, **kwargs):
        return [error(*args, **kwargs)]

    def _check_missing_label_categories(self, stats):
        validation_reports = []

        if len(stats["label_distribution"]["defined_labels"]) == 0:
            validation_reports += self._generate_validation_report(
                MissingLabelCategories, Severity.error
            )

        return validation_reports

    def _check_missing_annotation(self, stats):
        validation_reports = []

        items_missing = stats["items_missing_annotation"]
        for item_id, item_subset in items_missing:
            validation_reports += self._generate_validation_report(
                MissingAnnotation, Severity.warning, item_id, item_subset, self.str_ann_type
            )

        return validation_reports

    def _check_missing_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_missing_attr = attr_dets["items_missing_attribute"]
        for item_id, item_subset in items_missing_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                MissingAttribute, Severity.warning, item_id, *details
            )

        return validation_reports

    def _check_undefined_label(self, stats):
        validation_reports = []

        undefined_label_dist = stats["label_distribution"]["undefined_labels"]
        for label_name, label_stats in undefined_label_dist.items():
            for item_id, item_subset in label_stats["items_with_undefined_label"]:
                details = (item_subset, label_name)
                validation_reports += self._generate_validation_report(
                    UndefinedLabel, Severity.error, item_id, *details
                )

        return validation_reports

    def _check_undefined_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []

        items_with_undefined_attr = attr_dets["items_with_undefined_attr"]
        for item_id, item_subset in items_with_undefined_attr:
            details = (item_subset, label_name, attr_name)
            validation_reports += self._generate_validation_report(
                UndefinedAttribute, Severity.error, item_id, *details
            )

        return validation_reports

    def _check_label_defined_but_not_found(self, stats):
        validation_reports = []
        count_by_defined_labels = stats["label_distribution"]["defined_labels"]
        labels_not_found = [
            label_name for label_name, count in count_by_defined_labels.items() if count == 0
        ]

        for label_name in labels_not_found:
            validation_reports += self._generate_validation_report(
                LabelDefinedButNotFound, Severity.warning, label_name
            )

        return validation_reports

    def _check_attribute_defined_but_not_found(self, label_name, attr_stats):
        validation_reports = []
        attrs_not_found = [
            attr_name
            for attr_name, attr_dets in attr_stats.items()
            if len(attr_dets["distribution"]) == 0
        ]

        for attr_name in attrs_not_found:
            details = (label_name, attr_name)
            validation_reports += self._generate_validation_report(
                AttributeDefinedButNotFound, Severity.warning, *details
            )

        return validation_reports

    def _check_only_one_label(self, stats):
        validation_reports = []
        count_by_defined_labels = stats["label_distribution"]["defined_labels"]
        labels_found = [
            label_name for label_name, count in count_by_defined_labels.items() if count > 0
        ]

        if len(labels_found) == 1:
            validation_reports += self._generate_validation_report(
                OnlyOneLabel, Severity.info, labels_found[0]
            )

        return validation_reports

    def _check_only_one_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        values = list(attr_dets["distribution"].keys())

        if len(values) == 1:
            details = (label_name, attr_name, values[0])
            validation_reports += self._generate_validation_report(
                OnlyOneAttributeValue, Severity.info, *details
            )

        return validation_reports

    def _check_few_samples_in_label(self, stats):
        validation_reports = []
        thr = self.few_samples_thr

        defined_label_dist = stats["label_distribution"]["defined_labels"]
        labels_with_few_samples = [
            (label_name, count)
            for label_name, count in defined_label_dist.items()
            if 0 < count <= thr
        ]

        for label_name, count in labels_with_few_samples:
            validation_reports += self._generate_validation_report(
                FewSamplesInLabel, Severity.info, label_name, count
            )

        return validation_reports

    def _check_few_samples_in_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        thr = self.few_samples_thr

        attr_values_with_few_samples = [
            (attr_value, count)
            for attr_value, count in attr_dets["distribution"].items()
            if count <= thr
        ]

        for attr_value, count in attr_values_with_few_samples:
            details = (label_name, attr_name, attr_value, count)
            validation_reports += self._generate_validation_report(
                FewSamplesInAttribute, Severity.info, *details
            )

        return validation_reports

    def _check_imbalanced_labels(self, stats):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        defined_label_dist = stats["label_distribution"]["defined_labels"]
        count_by_defined_labels = [count for label, count in defined_label_dist.items()]

        if len(count_by_defined_labels) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_labels)
        count_min = np.min(count_by_defined_labels)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= thr:
            validation_reports += self._generate_validation_report(ImbalancedLabels, Severity.info)

        return validation_reports

    def _check_imbalanced_attribute(self, label_name, attr_name, attr_dets):
        validation_reports = []
        thr = self.imbalance_ratio_thr

        count_by_defined_attr = list(attr_dets["distribution"].values())
        if len(count_by_defined_attr) == 0:
            return validation_reports

        count_max = np.max(count_by_defined_attr)
        count_min = np.min(count_by_defined_attr)
        balance = count_max / count_min if count_min > 0 else float("inf")
        if balance >= thr:
            validation_reports += self._generate_validation_report(
                ImbalancedAttribute, Severity.info, label_name, attr_name
            )

        return validation_reports
