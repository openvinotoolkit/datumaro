# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from typing import Dict, List

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset import IDataset


class Severity(Enum):
    warning = auto()
    error = auto()


class TaskType(Enum):
    classification = auto()
    detection = auto()
    segmentation = auto()


class Validator(CliPlugin):
    def validate(self, dataset: IDataset) -> Dict:
        """
        Returns the validation results of a dataset based on task type.

        Args:
            dataset (IDataset): Dataset to be validated

        Raises:
            ValueError

        Returns:
            validation_results (dict):
                Dict with validation statistics, reports and summary.
        """

        validation_results = {}
        if not isinstance(dataset, IDataset):
            raise TypeError("Invalid dataset type '%s'" % type(dataset))

        # generate statistics
        stats = self.compute_statistics(dataset)
        validation_results["statistics"] = stats

        # generate validation reports and summary
        reports = self.generate_reports(stats)
        reports = list(map(lambda r: r.to_dict(), reports))

        summary = {
            "errors": sum(map(lambda r: r["severity"] == "error", reports)),
            "warnings": sum(map(lambda r: r["severity"] == "warning", reports)),
        }

        validation_results["validation_reports"] = reports
        validation_results["summary"] = summary

        return validation_results

    def compute_statistics(self, dataset: IDataset) -> Dict:
        """
        Computes statistics of the dataset based on task type.

        Args:
            dataset (IDataset): a dataset to be validated

        Returns:
            stats (dict): A dict object containing statistics of the dataset.
        """
        raise NotImplementedError("Must be implemented in a subclass")

    def generate_reports(self, stats: Dict) -> List[Dict]:
        raise NotImplementedError("Must be implemented in a subclass")
