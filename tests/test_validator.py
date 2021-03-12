# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest import TestCase

from assets.validator_test_data.datasets import (
    dataset_main,
    dataset_without_label_categories,
    dataset_w_only_one_label
)
from assets.validator_test_data.stats import (
    classification_stats_main, detection_stats_main,
    stats_without_label_categories, stats_w_only_one_label
)
from assets.validator_test_data.validation_results import (
    classification_summary, classification_val_reports,
    detection_summary, detection_val_reports
)
from datumaro.components.validator import (
    ClassificationValidator, DetectionValidator,
    Severity, validate_annotations, _Validator
)


class TestBaseValidator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = _Validator()

    def test_generate_validation_report_without_item_id(self):
        validation_item = 'TEST'
        description = 'unittest'
        severity = Severity.error

        actual = self.validator._generate_validation_report(
            validation_item, description, severity)
        expected = [{
            'anomaly_type': validation_item,
            'description': description,
            'severity': severity.name,
        }]

        self.assertEqual(expected, actual)

    def test_generate_validation_report_with_item_id(self):
        validation_item = 'TEST'
        description = 'unittest'
        severity = Severity.error
        item_id = 'test_id'

        actual = self.validator._generate_validation_report(
            validation_item, description, severity, item_id)
        expected = [{
            'anomaly_type': validation_item,
            'description': description,
            'severity': severity.name,
            'item_id': item_id
        }]

        self.assertEqual(expected, actual)

    def test_generate_reports(self):
        with self.assertRaisesRegex(
            NotImplementedError, 'Should be implemented in a subclass.'):
            self.validator.generate_reports({})

    def test_check_missing_attribute(self):
        stats = classification_stats_main
        label_name = 'label_0'
        attr_name = 'x'
        defined_attr_dist =\
             stats['attribute_distribution']['defined_attributes']
        attr_dets = defined_attr_dist[label_name][attr_name]
        description = f"DatasetItem needs the attribute '{attr_name}' " \
            f"for the label '{label_name}'."

        actual = self.validator._check_missing_attribute(
            label_name, attr_name, attr_dets)
        expected = [{
            'anomaly_type': 'MissingAttribute',
            'description': description,
            'severity': 'warning',
            'item_id': '4'
        }]

        self.assertEqual(expected, actual)

    def test_check_undefined_label(self):
        stats = classification_stats_main
        label_name = 4
        label_dist = stats['label_distribution']
        label_stats = label_dist['undefined_labels'][label_name]
        description = f"DatasetItem has the label '{label_name}' which " \
            "is not defined in LabelCategories."

        actual = self.validator._check_undefined_label(
            label_name, label_stats)
        expected = [{
            'anomaly_type': 'UndefinedLabel',
            'description': description,
            'severity': 'error',
            'item_id': '5'
        }]

        self.assertEqual(expected, actual)

    def test_check_undefined_attribute(self):
        stats = classification_stats_main
        label_name = 'label_1'
        attr_name = 'c'
        attr_dist = stats['attribute_distribution']
        undefined_attr_dist = attr_dist['undefined_attributes']
        attr_dets = undefined_attr_dist[label_name][attr_name]
        description = f"DatasetItem has the attribute '{attr_name}' for the " \
            f"label '{label_name}' which is not defined in LabelCategories."

        actual = self.validator._check_undefined_attribute(
            label_name, attr_name, attr_dets)
        expected = [{
            'anomaly_type': 'UndefinedAttribute',
            'description': description,
            'severity': 'error',
            'item_id': '6'
        }]

        self.assertEqual(expected, actual)

    def test_check_label_defined_but_not_found(self):
        stats = classification_stats_main
        label_not_found = 'label_3'
        description = f"The label '{label_not_found}' is defined in " \
            "LabelCategories, but not found in the dataset."

        actual = self.validator._check_label_defined_but_not_found(stats)
        expected = [{
            'anomaly_type': 'LabelDefinedButNotFound',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_attribute_defined_but_not_found(self):
        stats = classification_stats_main
        label_name = 'label_0'
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        attr_stats = defined_attr_dist[label_name]
        description = f"The attribute 'x' for the label '{label_name}' is " \
            "defined in LabelCategories, but not found in the dataset."

        actual = self.validator._check_attribute_defined_but_not_found(
            label_name, attr_stats)
        expected = [{
            'anomaly_type': 'AttributeDefinedButNotFound',
            'description': description,
            'severity': 'warning',
        }]

        self.assertCountEqual(expected, actual)

    def test_check_only_one_label(self):
        stats = stats_w_only_one_label
        label_name = 'label_0'
        description = f"The dataset has only one label '{label_name}'."

        actual = self.validator._check_only_one_label(stats)
        expected = [{
            'anomaly_type': 'OnlyOneLabel',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_only_one_attribute_value(self):
        stats = classification_stats_main
        label_name = 'label_0'
        attr_name = 'y'
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        attr_dets = defined_attr_dist[label_name][attr_name]
        description = "The dataset has the only attribute value " \
            f"'4' for the attribute '{attr_name}' for the label " \
            f"'{label_name}'."

        actual = self.validator._check_only_one_attribute_value(
            label_name, attr_name, attr_dets)
        expected = [{
            'anomaly_type': 'OnlyOneAttributeValue',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_few_samples_in_label(self):
        stats = classification_stats_main
        label_name = 'label_0'
        description = f"The number of samples in the label '{label_name}'" \
            f" might be too low. Found '1' samples."

        actual = self.validator._check_few_samples_in_label(stats, 2)
        expected = [{
            'anomaly_type': 'FewSamplesInLabel',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_few_samples_in_attribute(self):
        stats = classification_stats_main
        label_name = 'label_0'
        attr_name = 'y'
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        attr_dets = defined_attr_dist[label_name][attr_name]
        description = "The number of samples for attribute = value " \
            f"'{attr_name} = 4' for the label '{label_name}' " \
            f"might be too low. Found '1' samples."

        actual = self.validator._check_few_samples_in_attribute(
            label_name, attr_name, attr_dets, 2)
        expected = [{
            'anomaly_type': 'FewSamplesInAttribute',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_imbalanced_labels(self):
        stats = classification_stats_main
        description = 'There is an imbalance in the label distribution.'

        actual = self.validator._check_imbalanced_labels(stats, 5)
        expected = [{
            'anomaly_type': 'ImbalancedLabels',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_imbalanced_attribute(self):
        stats = classification_stats_main
        label_name = 'label_2'
        attr_name = 'x'
        attr_dist = stats['attribute_distribution']
        defined_attr_dist = attr_dist['defined_attributes']
        attr_dets = defined_attr_dist[label_name][attr_name]
        description = "There is an imbalance in the distribution of attribute" \
            f" '{attr_name}' for the label '{label_name}'."

        actual = self.validator._check_imbalanced_attribute(
            label_name, attr_name, attr_dets, 2)
        expected = [{
            'anomaly_type': 'ImbalancedAttribute',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)


class TestClassificationValidator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = ClassificationValidator()

    def test_compute_label_statistics(self):
        actual = self.validator.compute_statistics(dataset_main)
        expected = classification_stats_main

        self.assertEqual(expected, actual)

    def test_compute_label_statistics_missing_label_categories(self):
        actual = self.validator.compute_statistics(
            dataset_without_label_categories)
        expected = stats_without_label_categories

        self.assertEqual(expected, actual)

    def test_compute_label_statistics_with_only_one_label(self):
        actual = self.validator.compute_statistics(dataset_w_only_one_label)
        expected = stats_w_only_one_label

        self.assertEqual(expected, actual)

    def test_check_missing_label_categories(self):
        stats = stats_without_label_categories
        description = "'LabelCategories(...)' should be defined" \
            "to validate a dataset."

        actual = self.validator._check_missing_label_categories(stats)
        expected = [{
            'anomaly_type': 'MissingLabelCategories',
            'description': description,
            'severity': 'error',
        }]

        self.assertEqual(expected, actual)

    def test_check_missing_label_annotation(self):
        stats = classification_stats_main
        description = 'DatasetItem needs a Label(...) annotation, ' \
            'but not found.'

        actual = self.validator._check_missing_label_annotation(stats)
        expected = [{
            'anomaly_type': 'MissingLabelAnnotation',
            'description': description,
            'severity': 'warning',
            'item_id': '3',
        }]

        self.assertEqual(expected, actual)

    def test_check_multi_label_annotations(self):
        stats = classification_stats_main
        description = 'DatasetItem needs a single Label(...) annotation ' \
            'but multiple annotations are found.'

        actual = self.validator._check_multi_label_annotations(stats)
        expected = [{
            'anomaly_type': 'MultiLabelAnnotations',
            'description': description,
            'severity': 'error',
            'item_id': '4',
        }]

        self.assertEqual(expected, actual)

    def test_generate_reports(self):
        stats = classification_stats_main

        actual_reports = self.validator.generate_reports(stats)
        expected_reports = classification_val_reports

        self.assertCountEqual(expected_reports, actual_reports)


class TestDetectionValidator(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.validator = DetectionValidator()

    def test_compute_bbox_statistics(self):
        actual = self.validator.compute_statistics(dataset_main)
        expected = detection_stats_main

        self.assertEqual(expected, actual)

    def test_check_imbalanced_bbox_dist_in_label(self):
        stats = detection_stats_main
        label_name = 'label_0'
        bbox_dist_by_label = stats['bbox_distribution_in_label']
        bbox_label_stats = bbox_dist_by_label[label_name]
        description = "Values of bbox 'long' are not evenly distributed for " \
            "'label_0' label."

        actual = self.validator._check_imbalanced_bbox_dist_in_label(
            label_name, bbox_label_stats, 0.9, 0.1)
        expected = [{
            'anomaly_type': 'ImbalancedBboxDistInLabel',
            'description': description,
            'severity': 'warning',
        }]

        self.assertEqual(expected, actual)

    def test_check_imbalanced_bbox_dist_in_attr(self):
        stats = detection_stats_main
        label_name = 'label_2'
        attr_name = 'x'
        bbox_dist_by_attr = stats['bbox_distribution_in_attribute']
        bbox_attr_label = bbox_dist_by_attr[label_name]
        bbox_attr_stats = bbox_attr_label[attr_name]
        imbalanced_attr_value_props = [(1, 'height'), (1, 'long'), (3, 'y')]
        descriptions = [
            f"Values of bbox '{prop}' are not evenly " \
            f"distributed for '{attr_name}' = '{attr_value}' for " \
            f"the '{label_name}' label." \
            for attr_value, prop in imbalanced_attr_value_props
        ]

        actual = self.validator._check_imbalanced_bbox_dist_in_attr(
            label_name, attr_name, bbox_attr_stats, 0.9, 0.1)
        expected = [
            {
                'anomaly_type': 'ImbalancedBboxDistInAttribute',
                'description': d,
                'severity': 'warning',
            } for d in descriptions
        ]

        self.assertEqual(expected, actual)

    def test_check_missing_bbox_annotation(self):
        stats = detection_stats_main
        description = 'DatasetItem needs one or more Bbox(...) annotation, ' \
            'but not found.'

        actual = self.validator._check_missing_bbox_annotation(stats)
        expected = [{
            'anomaly_type': 'MissingBboxAnnotation',
            'description': description,
            'severity': 'warning',
            'item_id': '3'
        }]

        self.assertEqual(expected, actual)

    def test_check_negative_length(self):
        stats = detection_stats_main
        description = "Bbox annotation '2' in the DatasetItem should have a " \
            "positive value of 'height' but got '0'."

        actual = self.validator._check_negative_length(stats)
        expected = [{
            'anomaly_type': 'NegativeLength',
            'description': description,
            'severity': 'error',
            'item_id': '10'
        }]

        self.assertEqual(expected, actual)

    def test_check_invalid_value(self):
        stats = detection_stats_main
        description = "Bbox annotation '2' in the DatasetItem has an inf or " \
            "a NaN value of bbox 'ratio(w/h)'."

        actual = self.validator._check_invalid_value(stats)
        expected = [{
            'anomaly_type': 'InvalidValue',
            'description': description,
            'severity': 'error',
            'item_id': '10'
        }]

        self.assertEqual(expected, actual)

    def test_check_far_from_label_mean(self):
        stats = detection_stats_main
        label_name = 'label_2'
        bbox_dist_by_label = stats['bbox_distribution_in_label']
        bbox_label_stats = bbox_dist_by_label[label_name]
        far_from_label_mean = [
            (1, 'ratio(w/h)', 1.52, 4.0, '9'),
            (1, 'long', 3.57, 2, '15'),
        ]
        descriptions = [
            (
                f"Bbox annotation '{ann_id}' in " \
                f"the DatasetItem has a value of Bbox '{prop}' that " \
                "is too far from the label average. (mean of " \
                f"'{label_name}' label: {mean}, got '{val}').",
                item_id
            ) for ann_id, prop, mean, val, item_id in far_from_label_mean
        ]

        actual = self.validator._check_far_from_label_mean(
            label_name, bbox_label_stats)
        expected = [
            {
                'anomaly_type': 'FarFromLabelMean',
                'description': d,
                'severity': 'warning',
                'item_id': item_id
            } for d, item_id in descriptions
        ]

        self.assertEqual(expected, actual)

    def test_check_far_from_attr_mean(self):
        stats = detection_stats_main
        label_name = 'label_1'
        attr_name = 'x'
        bbox_dist_by_attr = stats['bbox_distribution_in_attribute']
        bbox_attr_label = bbox_dist_by_attr[label_name]
        bbox_attr_stats = bbox_attr_label[attr_name]
        description = "Bbox annotation '2' in the " \
            "DatasetItem has a value of Bbox 'y' that " \
            "is too far from the attribute average. (mean of " \
            f"'{attr_name}' = '3' for the " \
            f"'{label_name}' label: 16667.5, got '100000')."

        actual = self.validator._check_far_from_attr_mean(
            label_name, attr_name, bbox_attr_stats)
        expected = [{
            'anomaly_type': 'FarFromAttrMean',
            'description': description,
            'severity': 'warning',
            'item_id': '14'
        }]

        self.assertEqual(expected, actual)

    def test_generate_reports(self):
        stats = detection_stats_main

        actual_reports = self.validator.generate_reports(stats)
        expected_reports = detection_val_reports

        self.assertCountEqual(expected_reports, actual_reports)


class TestValidateAnnotations(TestCase):
    def test_validate_annotations_classification(self):
        actual = validate_annotations(dataset_main, 'classification')
        expected = {
            'statistics': classification_stats_main,
            'validation_reports': classification_val_reports,
            'summary': classification_summary
        }

        self.assertCountEqual(expected, actual)
        self.assertEqual(classification_stats_main, actual['statistics'])
        self.assertCountEqual(
            classification_val_reports, actual['validation_reports'])
        self.assertEqual(classification_summary, actual['summary'])

    def test_validate_annotations_detection(self):
        actual = validate_annotations(dataset_main, 'detection')
        expected = {
            'statistics': detection_stats_main,
            'validation_reports': detection_val_reports,
            'summary': detection_summary
        }

        self.assertCountEqual(expected, actual)
        self.assertEqual(detection_stats_main, actual['statistics'])
        self.assertCountEqual(
            detection_val_reports, actual['validation_reports'])
        self.assertEqual(detection_summary, actual['summary'])

    def test_validate_annotations_invalid_task_type(self):
        with self.assertRaisesRegex(ValueError, 'Invalid task type.'):
            validate_annotations(dataset_main, 'INVALID')

    def test_validate_annotations_invalid_dataset_type(self):
        with self.assertRaisesRegex(ValueError, 'Invalid Dataset type.'):
            validate_annotations({}, 'classification')
