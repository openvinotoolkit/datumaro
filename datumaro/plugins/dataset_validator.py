# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import numpy as np
from enum import Enum
from typing import Union

from datumaro.components.validator import (TaskType, Validator,
    ClassificationValidator, DetectionValidator, SegmentationValidator)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset import IDataset
from datumaro.util import parse_str_enum_value

from datumaro.cli.util import MultilineFormatter
from datumaro.cli.util.project import generate_next_file_name, load_project

TaskType = Enum('TaskType', ['classification', 'detection', 'segmentation'])

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