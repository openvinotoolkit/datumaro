# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import Counter
from unittest import TestCase
import numpy as np

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.errors import (MissingLabelCategories,
    MissingLabelAnnotation, MultiLabelAnnotations, MissingAttribute,
    UndefinedLabel, UndefinedAttribute, LabelDefinedButNotFound,
    AttributeDefinedButNotFound, OnlyOneLabel, FewSamplesInLabel,
    FewSamplesInAttribute, ImbalancedLabels, ImbalancedAttribute,
    ImbalancedBboxDistInLabel, ImbalancedBboxDistInAttribute,
    MissingBboxAnnotation, NegativeLength, InvalidValue, FarFromLabelMean,
    FarFromAttrMean, OnlyOneAttributeValue)
from datumaro.components.extractor import Bbox, Label
from datumaro.components.validator import (ClassificationValidator,
    DetectionValidator, TaskType, validate_annotations, _Validator)


class TestValidatorTemplate(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset.from_iterable([
            DatasetItem(id=1, image=np.ones((5, 5, 3)), annotations=[
                Label(1, id=0, attributes={'a': 1, 'b': 7, }),
                Bbox(1, 2, 3, 4, id=1, label=0, attributes={
                    'a': 1, 'b': 2,
                }),
            ]),
            DatasetItem(id=2, image=np.ones((2, 4, 3)), annotations=[
                Label(2, id=0, attributes={'a': 2, 'b': 2, }),
                Bbox(2, 3, 1, 4, id=1, label=0, attributes={
                    'a': 1, 'b': 1,
                }),
            ]),
            DatasetItem(id=3),
            DatasetItem(id=4, image=np.ones((2, 4, 3)), annotations=[
                Label(0, id=0, attributes={'b': 4, }),
                Label(1, id=1, attributes={'a': 11, 'b': 7, }),
                Bbox(1, 3, 2, 4, id=2, label=0, attributes={
                    'a': 2, 'b': 1,
                }),
                Bbox(3, 1, 4, 2, id=3, label=0, attributes={
                    'a': 2, 'b': 2,
                }),
            ]),
            DatasetItem(id=5, image=np.ones((2, 4, 3)), annotations=[
                Label(0, id=0, attributes={'a': 20, 'b': 10, }),
                Bbox(1, 2, 3, 4, id=1, label=1, attributes={
                    'a': 1, 'b': 1,
                }),
            ]),
            DatasetItem(id=6, image=np.ones((2, 4, 3)), annotations=[
                Label(1, id=0, attributes={'a': 11, 'b': 2, 'c': 3, }),
                Bbox(2, 3, 4, 1, id=1, label=1, attributes={
                    'a': 2, 'b': 2,
                }),
            ]),
            DatasetItem(id=7, image=np.ones((2, 4, 3)), annotations=[
                Label(1, id=0, attributes={'a': 1, 'b': 2, 'c': 5, }),
                Bbox(1, 2, 3, 4, id=1, label=2, attributes={
                    'a': 1, 'b': 2,
                }),
            ]),
            DatasetItem(id=8, image=np.ones((2, 4, 3)), annotations=[
                Label(2, id=0, attributes={'a': 7, 'b': 9, 'c': 5, }),
                Bbox(2, 1, 3, 4, id=1, label=2, attributes={
                    'a': 2, 'b': 1,
                }),
            ]),
        ], categories=[[f'label_{i}', None, {'a', 'b', }]
            for i in range(2)])


class TestBaseValidator(TestValidatorTemplate):
    @classmethod
    def setUpClass(cls):
        cls.validator = _Validator(TaskType.classification)

    def test_generate_reports(self):
        with self.assertRaises(NotImplementedError):
            self.validator.generate_reports({})

    def test_check_missing_label_categories(self):
        stats = {
            'label_distribution': {
                'defined_labels': {}
            }
        }

        actual_reports = self.validator._check_missing_label_categories(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingLabelCategories)

    def test_check_missing_attribute(self):
        label_name = 'unit'
        attr_name = 'test'
        attr_dets = {
            'items_missing_attribute': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_missing_attribute(
            label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingAttribute)

    def test_check_undefined_label(self):
        label_name = 'unittest'
        label_stats = {
            'items_with_undefined_label': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_undefined_label(
            label_name, label_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], UndefinedLabel)

    def test_check_undefined_attribute(self):
        label_name = 'unit'
        attr_name = 'test'
        attr_dets = {
            'items_with_undefined_attr': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_undefined_attribute(
            label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], UndefinedAttribute)

    def test_check_label_defined_but_not_found(self):
        stats = {
            'label_distribution': {
                'defined_labels': {
                    'unittest': 0
                }
            }
        }

        actual_reports = self.validator._check_label_defined_but_not_found(
            stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], LabelDefinedButNotFound)

    def test_check_attribute_defined_but_not_found(self):
        label_name = 'unit'
        attr_stats = {
            'test': {
                'distribution': {}
            }
        }

        actual_reports = self.validator._check_attribute_defined_but_not_found(
            label_name, attr_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], AttributeDefinedButNotFound)

    def test_check_only_one_label(self):
        stats = {
            'label_distribution': {
                'defined_labels': {
                    'unit': 1,
                    'test': 0
                }
            }
        }

        actual_reports = self.validator._check_only_one_label(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], OnlyOneLabel)

    def test_check_only_one_attribute_value(self):
        label_name = 'unit'
        attr_name = 'test'
        attr_dets = {
            'distribution': {
                'mock': 1
            }
        }

        actual_reports = self.validator._check_only_one_attribute_value(
            label_name, attr_name, attr_dets)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], OnlyOneAttributeValue)

    def test_check_few_samples_in_label(self):
        with self.subTest('Few Samples'):
            stats = {
                'label_distribution': {
                    'defined_labels': {
                        'unit': self.validator.DEFAULT_FEW_SAMPLES
                    }
                }
            }

            actual_reports = self.validator._check_few_samples_in_label(stats)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], FewSamplesInLabel)

        with self.subTest('No Few Samples Warning'):
            stats = {
                'label_distribution': {
                    'defined_labels': {
                        'unit': self.validator.DEFAULT_FEW_SAMPLES + 1
                    }
                }
            }

            actual_reports = self.validator._check_few_samples_in_label(stats)

            self.assertTrue(len(actual_reports) == 0)

    def test_check_few_samples_in_attribute(self):
        label_name = 'unit'
        attr_name = 'test'

        with self.subTest('Few Samples'):
            attr_dets = {
                'distribution': {
                    'mock': self.validator.DEFAULT_FEW_SAMPLES
                }
            }

            actual_reports = self.validator._check_few_samples_in_attribute(
                label_name, attr_name, attr_dets)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], FewSamplesInAttribute)

        with self.subTest('No Few Samples Warning'):
            attr_dets = {
                'distribution': {
                    'mock': self.validator.DEFAULT_FEW_SAMPLES + 1
                }
            }

            actual_reports = self.validator._check_few_samples_in_attribute(
                label_name, attr_name, attr_dets)

            self.assertTrue(len(actual_reports) == 0)

    def test_check_imbalanced_labels(self):
        with self.subTest('Imbalance'):
            stats = {
                'label_distribution': {
                    'defined_labels': {
                        'unit': self.validator.DEFAULT_IMBALANCE_RATIO,
                        'test': 1
                    }
                }
            }

            actual_reports = self.validator._check_imbalanced_labels(stats)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], ImbalancedLabels)

        with self.subTest('No Imbalance Warning'):
            stats = {
                'label_distribution': {
                    'defined_labels': {
                        'unit': self.validator.DEFAULT_IMBALANCE_RATIO - 1,
                        'test': 1
                    }
                }
            }

            actual_reports = self.validator._check_imbalanced_labels(stats)

            self.assertTrue(len(actual_reports) == 0)

    def test_check_imbalanced_attribute(self):
        label_name = 'unit'
        attr_name = 'test'

        with self.subTest('Imbalance'):
            attr_dets = {
                'distribution': {
                    'mock': self.validator.DEFAULT_IMBALANCE_RATIO,
                    'mock_1': 1
                }
            }

            actual_reports = self.validator._check_imbalanced_attribute(
                label_name, attr_name, attr_dets)

            self.assertTrue(len(actual_reports) == 1)
            self.assertIsInstance(actual_reports[0], ImbalancedAttribute)

        with self.subTest('No Imbalance Warning'):
            attr_dets = {
                'distribution': {
                    'mock': self.validator.DEFAULT_IMBALANCE_RATIO - 1,
                    'mock_1': 1
                }
            }

            actual_reports = self.validator._check_imbalanced_attribute(
                label_name, attr_name, attr_dets)

            self.assertTrue(len(actual_reports) == 0)


class TestClassificationValidator(TestValidatorTemplate):
    @classmethod
    def setUpClass(cls):
        cls.validator = ClassificationValidator()

    def test_check_missing_label_annotation(self):
        stats = {
            'items_missing_label': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_missing_label_annotation(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingLabelAnnotation)

    def test_check_multi_label_annotations(self):
        stats = {
            'items_with_multiple_labels': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_multi_label_annotations(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MultiLabelAnnotations)


class TestDetectionValidator(TestValidatorTemplate):
    @classmethod
    def setUpClass(cls):
        cls.validator = DetectionValidator()

    def test_check_imbalanced_bbox_dist_in_label(self):
        label_name = 'unittest'
        most = int(self.validator.DEFAULT_BBOX_IMBALANCE * 100)
        rest = 100 - most

        with self.subTest('Imbalanced'):
            bbox_label_stats = {
                'x': {
                    'histogram': {
                        'counts': [most, rest]
                    }
                }
            }
            reports = self.validator._check_imbalanced_bbox_dist_in_label(
                label_name, bbox_label_stats)

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedBboxDistInLabel)

        with self.subTest('No Imbalanced Warning'):
            bbox_label_stats = {
                'x': {
                    'histogram': {
                        'counts': [most - 1, rest]
                    }
                }
            }
            reports = self.validator._check_imbalanced_bbox_dist_in_label(
                label_name, bbox_label_stats)

            self.assertTrue(len(reports) == 0)

    def test_check_imbalanced_bbox_dist_in_attr(self):
        label_name = 'unit'
        attr_name = 'test'
        most = int(self.validator.DEFAULT_BBOX_IMBALANCE * 100)
        rest = 100 - most

        with self.subTest('Imbalanced'):
            bbox_attr_stats = {
                'mock': {
                    'x': {
                        'histogram': {
                            'counts': [most, rest]
                        }
                    }
                }
            }

            reports = self.validator._check_imbalanced_bbox_dist_in_attr(
                label_name, attr_name, bbox_attr_stats)

            self.assertTrue(len(reports) == 1)
            self.assertIsInstance(reports[0], ImbalancedBboxDistInAttribute)

        with self.subTest('No Imbalanced Warning'):
            bbox_attr_stats = {
                'mock': {
                    'x': {
                        'histogram': {
                            'counts': [most - 1, rest]
                        }
                    }
                }
            }

            reports = self.validator._check_imbalanced_bbox_dist_in_attr(
                label_name, attr_name, bbox_attr_stats)

            self.assertTrue(len(reports) == 0)

    def test_check_missing_bbox_annotation(self):
        stats = {
            'items_missing_bbox': [(1, 'unittest')]
        }

        actual_reports = self.validator._check_missing_bbox_annotation(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], MissingBboxAnnotation)

    def test_check_negative_length(self):
        stats = {
            'items_with_negative_length': {
                ('1', 'unittest'): {
                    1: {
                        'x': -1
                    }
                }
            }
        }

        actual_reports = self.validator._check_negative_length(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], NegativeLength)

    def test_check_invalid_value(self):
        stats = {
            'items_with_invalid_value': {
                ('1', 'unittest'): {
                    1: ['x']
                }
            }
        }

        actual_reports = self.validator._check_invalid_value(stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], InvalidValue)

    def test_check_far_from_label_mean(self):
        label_name = 'unittest'
        bbox_label_stats = {
            'w': {
                'items_far_from_mean': {
                    ('1', 'unittest'): {
                        1: 100
                    }
                },
                'mean': 0,
            }
        }

        actual_reports = self.validator._check_far_from_label_mean(
            label_name, bbox_label_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromLabelMean)

    def test_check_far_from_attr_mean(self):
        label_name = 'unit'
        attr_name = 'test'
        bbox_attr_stats = {
            'mock': {
                'w': {
                    'items_far_from_mean': {
                        ('1', 'unittest'): {
                            1: 100
                        }
                    },
                    'mean': 0,
                }
            }
        }

        actual_reports = self.validator._check_far_from_attr_mean(
            label_name, attr_name, bbox_attr_stats)

        self.assertTrue(len(actual_reports) == 1)
        self.assertIsInstance(actual_reports[0], FarFromAttrMean)


class TestValidateAnnotations(TestValidatorTemplate):
    def test_validate_annotations_classification(self):
        actual_results = validate_annotations(self.dataset, 'classification')

        with self.subTest('Test of statistics', i=0):
            actual_stats = actual_results['statistics']
            self.assertEqual(actual_stats['total_label_count'], 8)
            self.assertEqual(len(actual_stats['items_missing_label']), 1)
            self.assertEqual(len(actual_stats['items_with_multiple_labels']), 1)

            label_dist = actual_stats['label_distribution']
            defined_label_dist = label_dist['defined_labels']
            self.assertEqual(len(defined_label_dist), 2)
            self.assertEqual(sum(defined_label_dist.values()), 6)

            undefined_label_dist = label_dist['undefined_labels']
            undefined_label_stats = undefined_label_dist[2]
            self.assertEqual(len(undefined_label_dist), 1)
            self.assertEqual(undefined_label_stats['count'], 2)
            self.assertEqual(
                len(undefined_label_stats['items_with_undefined_label']), 2)

            attr_stats = actual_stats['attribute_distribution']
            defined_attr_dets = attr_stats['defined_attributes']['label_0']['a']
            self.assertEqual(
                len(defined_attr_dets['items_missing_attribute']), 1)
            self.assertEqual(defined_attr_dets['distribution'], {'20': 1})

            undefined_attr_dets = attr_stats['undefined_attributes'][2]['c']
            self.assertEqual(
                len(undefined_attr_dets['items_with_undefined_attr']), 1)
            self.assertEqual(undefined_attr_dets['distribution'], {'5': 1})

        with self.subTest('Test of validation reports', i=1):
            actual_reports = actual_results['validation_reports']
            report_types = [r['anomaly_type'] for r in actual_reports]
            report_count_by_type = Counter(report_types)

            self.assertEqual(len(actual_reports), 16)
            self.assertEqual(report_count_by_type['UndefinedAttribute'], 7)
            self.assertEqual(report_count_by_type['FewSamplesInAttribute'], 3)
            self.assertEqual(report_count_by_type['UndefinedLabel'], 2)
            self.assertEqual(report_count_by_type['MissingLabelAnnotation'], 1)
            self.assertEqual(report_count_by_type['MultiLabelAnnotations'], 1)
            self.assertEqual(report_count_by_type['OnlyOneAttributeValue'], 1)
            self.assertEqual(report_count_by_type['MissingAttribute'], 1)

        with self.subTest('Test of summary', i=2):
            actual_summary = actual_results['summary']
            expected_summary = {
                'errors': 10,
                'warnings': 6
            }

            self.assertEqual(actual_summary, expected_summary)

    def test_validate_annotations_detection(self):
        actual_results = validate_annotations(self.dataset, 'detection')

        with self.subTest('Test of statistics', i=0):
            actual_stats = actual_results['statistics']
            self.assertEqual(actual_stats['total_bbox_count'], 8)
            self.assertEqual(len(actual_stats['items_missing_bbox']), 1)
            self.assertEqual(actual_stats['items_with_negative_length'], {})
            self.assertEqual(actual_stats['items_with_invalid_value'], {})

            bbox_dist_by_label = actual_stats['bbox_distribution_in_label']
            label_prop_stats = bbox_dist_by_label['label_1']['width']
            self.assertEqual(label_prop_stats['items_far_from_mean'], {})
            self.assertEqual(label_prop_stats['mean'], 3.5)
            self.assertEqual(label_prop_stats['stdev'], 0.5)
            self.assertEqual(label_prop_stats['min'], 3.0)
            self.assertEqual(label_prop_stats['max'], 4.0)
            self.assertEqual(label_prop_stats['median'], 3.5)

            bbox_dist_by_attr = actual_stats['bbox_distribution_in_attribute']
            attr_prop_stats = bbox_dist_by_attr['label_0']['a']['1']['width']
            self.assertEqual(attr_prop_stats['items_far_from_mean'], {})
            self.assertEqual(attr_prop_stats['mean'], 2.0)
            self.assertEqual(attr_prop_stats['stdev'], 1.0)
            self.assertEqual(attr_prop_stats['min'], 1.0)
            self.assertEqual(attr_prop_stats['max'], 3.0)
            self.assertEqual(attr_prop_stats['median'], 2.0)

            bbox_dist_item = actual_stats['bbox_distribution_in_dataset_item']
            self.assertEqual(sum(bbox_dist_item.values()), 8)

        with self.subTest('Test of validation reports', i=1):
            actual_reports = actual_results['validation_reports']
            report_types = [r['anomaly_type'] for r in actual_reports]
            count_by_type = Counter(report_types)

            self.assertEqual(len(actual_reports), 45)
            self.assertEqual(count_by_type['ImbalancedBboxDistInAttribute'], 32)
            self.assertEqual(count_by_type['FewSamplesInAttribute'], 4)
            self.assertEqual(count_by_type['UndefinedAttribute'], 4)
            self.assertEqual(count_by_type['ImbalancedBboxDistInLabel'], 2)
            self.assertEqual(count_by_type['UndefinedLabel'], 2)
            self.assertEqual(count_by_type['MissingBboxAnnotation'], 1)

        with self.subTest('Test of summary', i=2):
            actual_summary = actual_results['summary']
            expected_summary = {
                'errors': 6,
                'warnings': 39
            }

            self.assertEqual(actual_summary, expected_summary)

    def test_validate_annotations_invalid_task_type(self):
        with self.assertRaises(ValueError):
            validate_annotations(self.dataset, 'INVALID')

    def test_validate_annotations_invalid_dataset_type(self):
        with self.assertRaises(TypeError):
            validate_annotations(object(), 'classification')
