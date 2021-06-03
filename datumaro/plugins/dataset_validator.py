# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Union

from datumaro.components.validator import (TaskType, Validator,
    ClassificationValidator, DetectionValidator, SegmentationValidator)
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset import IDataset
from datumaro.util import parse_str_enum_value

class DatasetValidator(Validator, CliPlugin):

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
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
        return parser

    def validate_annotations(dataset: IDataset, task_type: Union[str, TaskType], **extra_args):
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

        few_samples_thr = extra_args['few_samples_thr']
        imbalance_ratio_thr = extra_args['imbalance_ratio_thr']
        far_from_mean_thr = extra_args['far_from_mean_thr']
        dominance_ratio_thr = extra_args['dominance_ratio_thr']
        topk_bins = extra_args['topk_bins']

        validation_results = {}

        task_type = parse_str_enum_value(task_type, TaskType)
        if task_type == TaskType.classification:
            validator = ClassificationValidator(few_samples_thr=few_samples_thr,
                imbalance_ratio_thr=imbalance_ratio_thr,
                far_from_mean_thr=far_from_mean_thr,
                dominance_ratio_thr=dominance_ratio_thr,
                topk_bins=topk_bins)
        elif task_type == TaskType.detection:
            validator = DetectionValidator(few_samples_thr=few_samples_thr,
                imbalance_ratio_thr=imbalance_ratio_thr,
                far_from_mean_thr=far_from_mean_thr,
                dominance_ratio_thr=dominance_ratio_thr,
                topk_bins=topk_bins)
        elif task_type == TaskType.segmentation:
            validator = SegmentationValidator(few_samples_thr=few_samples_thr,
                imbalance_ratio_thr=imbalance_ratio_thr,
                far_from_mean_thr=far_from_mean_thr,
                dominance_ratio_thr=dominance_ratio_thr,
                topk_bins=topk_bins)

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