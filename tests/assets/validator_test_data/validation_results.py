# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

classification_val_reports = [
    {
        'anomaly_type': 'MissingLabelAnnotation',
        'description': 'DatasetItem needs a Label(...) annotation, ' \
            'but not found.',
        'severity': 'warning',
        'item_id': '3'
    },
    {
        'anomaly_type': 'MultiLabelAnnotations',
        'description': 'DatasetItem needs a single Label(...) ' \
            'annotation but multiple annotations are found.',
        'severity': 'error',
        'item_id': '4'
    },
    {
        'anomaly_type': 'LabelDefinedButNotFound',
        'description': "The label 'label_3' is defined in " \
            "LabelCategories, but not found in the dataset.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FewSamplesInLabel',
        'description': "The number of samples in the label 'label_0' " \
            "might be too low. Found '1' samples.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'ImbalancedLabels',
        'description': 'There is an imbalance in the label ' \
            'distribution.',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FewSamplesInAttribute',
        'description': "The number of samples for attribute = value " \
            "'z = 5' for the label 'label_0' might be too low. " \
            "Found '1' samples.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'OnlyOneAttributeValue',
        'description': "The dataset has the only attribute value '5' " \
            "for the attribute 'z' for the label 'label_0'.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'MissingAttribute',
        'description': "DatasetItem needs the attribute 'x' " \
            "for the label 'label_0'.",
        'severity': 'warning',
        'item_id': '4'
    },
    {
        'anomaly_type': 'AttributeDefinedButNotFound',
        'description': "The attribute 'x' for the label 'label_0' is " \
            "defined in LabelCategories, but not found in the dataset.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FewSamplesInAttribute',
        'description': "The number of samples for attribute = value " \
            "'y = 4' for the label 'label_0' might be too low. " \
            "Found '1' samples.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'OnlyOneAttributeValue',
        'description': "The dataset has the only attribute value '4' " \
            "for the attribute 'y' for the label 'label_0'.",
        'severity': 'warning'
    },
    {
        'anomaly_type': 'UndefinedLabel',
        'description': "DatasetItem has the label '4' which is not " \
            "defined in LabelCategories.",
        'severity': 'error',
        'item_id': '5'
    },
    {
        'anomaly_type': 'UndefinedAttribute',
        'description': "DatasetItem has the attribute 'a' for the " \
            "label '4' which is not defined in LabelCategories.",
        'severity': 'error',
        'item_id': '5'
    },
    {
        'anomaly_type': 'UndefinedAttribute',
        'description': "DatasetItem has the attribute 'c' for the " \
            "label 'label_1' which is not defined in LabelCategories.",
        'severity': 'error',
        'item_id': '6'
    },
]

classification_summary = {
    'errors': 4,
    'warnings': 10
}

detection_val_reports = [
    {
        'anomaly_type': 'MissingBboxAnnotation',
        'description': 'DatasetItem needs one or more ' \
                    'Bbox(...) annotation, but not found.',
        'item_id': '3',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'LabelDefinedButNotFound',
        'description': "The label 'label_3' is defined in " \
                    'LabelCategories, but not found in the ' \
                    'dataset.',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'ImbalancedLabels',
        'description': 'There is an imbalance in the label ' \
                    'distribution.',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'NegativeLength',
        'description': "Bbox annotation '2' in the " \
                    'DatasetItem should have a positive ' \
                    "value of 'height' but got '0'.",
        'item_id': '10',
        'severity': 'error'
    },
    {
        'anomaly_type': 'InvalidValue',
        'description': "Bbox annotation '2' in the " \
                    'DatasetItem has an inf or a NaN value ' \
                    "of bbox 'ratio(w/h)'.",
        'item_id': '10',
        'severity': 'error'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '2' in the "
                    'DatasetItem has a value of Bbox ' \
                    "'ratio(w/h)' that is too far from the " \
                    "label average. (mean of 'label_0' " \
                    "label: 1.38, got '4.0').",
        'item_id': '2',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '1' in the " \
                    "DatasetItem has a value of Bbox 'x' " \
                    'that is too far from the label ' \
                    "average. (mean of 'label_1' label: " \
                    "1.9, got '4').",
        'item_id': '14',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '2' in the " \
                    "DatasetItem has a value of Bbox 'y' " \
                    'that is too far from the label ' \
                    "average. (mean of 'label_1' label: " \
                    "10001.6, got '100000').",
        'item_id': '14',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '1' in the " \
                    'DatasetItem has a value of Bbox ' \
                    "'ratio(w/h)' that is too far from the " \
                    "label average. (mean of 'label_1' " \
                    "label: 1.24, got '4.0').",
        'item_id': '6',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromAttrMean',
        'description': "Bbox annotation '2' in the " \
                    "DatasetItem has a value of Bbox 'y' " \
                    'that is too far from the attribute ' \
                    "average. (mean of 'x' = '3' for the " \
                    "'label_1' label: 16667.5, got '100000').",
        'item_id': '14',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromAttrMean',
        'description': "Bbox annotation '2' in the " \
                    "DatasetItem has a value of Bbox 'y' " \
                    'that is too far from the attribute ' \
                    "average. (mean of 'y' = '1' for the " \
                    "'label_1' label: 16667.5, got '100000').",
        'item_id': '14',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromAttrMean',
        'description': "Bbox annotation '2' in the " \
                    "DatasetItem has a value of Bbox 'y' " \
                    'that is too far from the attribute ' \
                    "average. (mean of 'z' = '2' for the " \
                    "'label_1' label: 16667.5, got '100000').",
        'item_id': '14',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '1' in the " \
                    'DatasetItem has a value of Bbox ' \
                    "'ratio(w/h)' that is too far from the " \
                    "label average. (mean of 'label_2' " \
                    "label: 1.52, got '4.0').",
        'item_id': '9',
        'severity': 'warning'
    },
    {
        'anomaly_type': 'FarFromLabelMean',
        'description': "Bbox annotation '1' in the " \
                    'DatasetItem has a value of Bbox ' \
                    "'long' that is too far from the label " \
                    "average. (mean of 'label_2' label: " \
                    "3.57, got '2').",
        'item_id': '15',
        'severity': 'warning'
    },
]

detection_summary = {
    'errors': 2,
    'warnings': 12
}
