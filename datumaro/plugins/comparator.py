# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from typing import Dict, List, Tuple

from attr import attrs
from texttable import Texttable

from datumaro.cli.util.project import generate_next_file_name
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.plugins.shift_analyzer import ShiftAnalyzer
from datumaro.util import dump_json_file


@attrs
class Comparator:
    """
    Class for comparing datasets and generating comparison reports.
    """

    @staticmethod
    def _extract_labels(dataset) -> set:
        """Extract labels from the dataset.

        Args:
            dataset: The dataset to extract labels from.

        Returns:
            A set of label names.
        """
        label_cat = dataset.categories().get(AnnotationType.label, LabelCategories())
        return set(c.name for c in label_cat)

    @staticmethod
    def _compute_statistics(dataset) -> Tuple[Dict, Dict]:
        """Compute image and annotation statistics.

        Args:
            dataset: The dataset to compute statistics for.

        Returns:
            Tuple of image statistics and annotation statistics.
        """
        image_stats = compute_image_statistics(dataset)
        ann_stats = compute_ann_statistics(dataset)
        return image_stats, ann_stats

    def _analyze_dataset(self, dataset) -> Tuple[str, set, Dict, Dict]:
        """Analyze the dataset to get labels, format, statistics.

        Args:
            dataset: The dataset to analyze.

        Returns:
            Tuple of dataset format, label names, image statistics, annotation statistics.
        """
        dataset_format = dataset.format
        dataset_labels = self._extract_labels(dataset)
        image_stats, ann_stats = self._compute_statistics(dataset)
        return dataset_format, dataset_labels, image_stats, ann_stats

    @staticmethod
    def _create_text_table(columns_align: List[str], columns_valign: List[str]) -> Texttable:
        """Create a text table with the given alignments.

        Args:
            columns_align: List of alignment values for each column.
            columns_valign: List of vertical alignment values for each column.

        Returns:
            A Texttable object.
        """
        table = Texttable()
        table.set_cols_align(columns_align)
        table.set_cols_valign(columns_valign)
        return table

    @staticmethod
    def _create_rows_and_dict(rows: List[List]) -> Tuple[List[List], Dict]:
        """Create rows for the text table and a dictionary of data.

        Args:
            rows: List of row data.

        Returns:
            Tuple of rows and data dictionary.
        """
        data_dict = {row[0]: row[1:] for row in rows[1:]}
        return rows, data_dict

    def _create_high_level_comparison_table(self, first_info, second_info) -> Tuple[str, Dict]:
        """Generate a high-level comparison table.

        Args:
            first_info: Tuple of first dataset information.
            second_info: Tuple of second dataset information.

        Returns:
            Tuple of the table as a string and a data dictionary.
        """
        first_format, first_labels, first_image_stats, first_ann_stats = first_info
        second_format, second_labels, second_image_stats, second_ann_stats = second_info

        table = self._create_text_table(["l", "l", "l"], ["m", "m", "m"])

        rows = [
            ["Field", "Source", "Target"],
            ["Format", first_format, second_format],
            ["Number of classes", len(first_labels), len(second_labels)],
            [
                "Intersect classes",
                ", ".join(sorted(list(first_labels.intersection(second_labels)))),
                ", ".join(sorted(list(first_labels.intersection(second_labels)))),
            ],
            ["Classes", ", ".join(sorted(first_labels)), ", ".join(sorted(second_labels))],
            [
                "Images count",
                first_image_stats["dataset"]["images count"],
                second_image_stats["dataset"]["images count"],
            ],
            [
                "Unique images count",
                first_image_stats["dataset"]["unique images count"],
                second_image_stats["dataset"]["unique images count"],
            ],
            [
                "Repeated images count",
                first_image_stats["dataset"]["repeated images count"],
                second_image_stats["dataset"]["repeated images count"],
            ],
            [
                "Annotations count",
                first_ann_stats["annotations count"],
                second_ann_stats["annotations count"],
            ],
            [
                "Unannotated images count",
                first_ann_stats["unannotated images count"],
                second_ann_stats["unannotated images count"],
            ],
        ]

        rows, data_dict = self._create_rows_and_dict(rows)
        table.add_rows(rows)
        return table.draw(), data_dict

    def _create_mid_level_comparison_table(self, first_info, second_info) -> Tuple[str, Dict]:
        """Generate a mid-level comparison table.

        Args:
            first_info: Tuple of first dataset information.
            second_info: Tuple of second dataset information.

        Returns:
            Tuple of the table as a string and a data dictionary.
        """
        _, _, first_image_stats, first_ann_stats = first_info
        _, _, second_image_stats, second_ann_stats = second_info

        table = self._create_text_table(["l", "l", "l"], ["m", "m", "m"])

        rows = [
            ["Field", "Source", "Target"],
        ]

        # sort by subset names
        subset_names = sorted(
            set(first_image_stats["subsets"].keys()).union(second_image_stats["subsets"].keys())
        )

        for subset_name in subset_names:
            first_subset_data = first_image_stats["subsets"].get(subset_name, {})
            second_subset_data = second_image_stats["subsets"].get(subset_name, {})
            mean_str_first = (
                ", ".join(f"{val:6.2f}" for val in first_subset_data.get("image mean", []))
                if "image mean" in first_subset_data
                else ""
            )
            std_str_first = (
                ", ".join(f"{val:6.2f}" for val in first_subset_data.get("image std", []))
                if "image std" in first_subset_data
                else ""
            )
            mean_str_second = (
                ", ".join(f"{val:6.2f}" for val in second_subset_data.get("image mean", []))
                if "image mean" in second_subset_data
                else ""
            )
            std_str_second = (
                ", ".join(f"{val:6.2f}" for val in second_subset_data.get("image std", []))
                if "image std" in second_subset_data
                else ""
            )
            rows.append([f"{subset_name} - Image Mean", mean_str_first, mean_str_second])
            rows.append([f"{subset_name} - Image Std", std_str_first, std_str_second])

        label_names = sorted(
            set(first_ann_stats["annotations"]["labels"]["distribution"].keys()).union(
                second_ann_stats["annotations"]["labels"]["distribution"].keys()
            )
        )

        for label_name in label_names:
            count_dist_first = first_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_dist_second = second_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_first, dist_first = count_dist_first if count_dist_first[0] != 0 else ["", ""]
            count_second, dist_second = count_dist_second if count_dist_second[0] != 0 else ["", ""]
            rows.append(
                [
                    f"Label - {label_name}",
                    f"imgs: {count_first}, percent: {dist_first:.4f}" if count_first != "" else "",
                    f"imgs: {count_second}, percent: {dist_second:.4f}"
                    if count_second != ""
                    else "",
                ]
            )

        rows, data_dict = self._create_rows_and_dict(rows)
        table.add_rows(rows)

        return table.draw(), data_dict

    def _create_low_level_comparison_table(self, first_dataset, second_dataset) -> Tuple[str, Dict]:
        """Generate a low-level comparison table.

        Args:
            first_dataset: First dataset to compare.
            second_dataset: Second dataset to compare.

        Returns:
            Tuple of the table as a string and a data dictionary.
        """
        shift_analyzer = ShiftAnalyzer()
        cov_shift = shift_analyzer.compute_covariate_shift([first_dataset, second_dataset])
        label_shift = shift_analyzer.compute_label_shift([first_dataset, second_dataset])

        table = self._create_text_table(["l", "l"], ["m", "m"])

        rows = [
            ["Field", "Value"],
            ["Covariate shift", cov_shift],
            ["Label shift", label_shift],
        ]

        rows, data_dict = self._create_rows_and_dict(rows)
        table.add_rows(rows)
        return table.draw(), data_dict

    def compare_datasets(self, first, second) -> Tuple[str, str, str, Dict]:
        """Compare two datasets and generate comparison reports.

        Args:
            first: First dataset to compare.
            second: Second dataset to compare.

        Returns:
            Tuple of high-level table, mid-level table, low-level table, comparison dictionary.
        """
        first_info = self._analyze_dataset(first)
        second_info = self._analyze_dataset(second)

        high_level_table, high_level_dict = self._create_high_level_comparison_table(
            first_info, second_info
        )
        mid_level_table, mid_level_dict = self._create_mid_level_comparison_table(
            first_info, second_info
        )
        low_level_table, low_level_dict = self._create_low_level_comparison_table(first, second)

        comparison_dict = dict(
            high_level=high_level_dict, mid_level=mid_level_dict, low_level=low_level_dict
        )

        print(f"High-level comparison:\n{high_level_table}\n")
        print(f"Mid-level comparison:\n{mid_level_table}\n")
        print(f"Low-level comparison:\n{low_level_table}\n")

        return high_level_table, mid_level_table, low_level_table, comparison_dict

    @staticmethod
    def save_compare_report(
        high_level_table, mid_level_table, low_level_table, comparison_dict, report_dir: str
    ) -> None:
        """Save the comparison report to JSON and text files.

        Args:
            high_level_table: High-level comparison table as a string.
            mid_level_table: Mid-level comparison table as a string.
            low_level_table: Low-level comparison table as a string.
            comparison_dict: Comparison dictionary.
            report_dir: Directory to save the report files.
        """
        os.makedirs(report_dir, exist_ok=True)
        json_output_file = osp.join(
            report_dir, generate_next_file_name("compare", ext=".json", basedir=report_dir)
        )
        txt_output_file = osp.join(
            report_dir, generate_next_file_name("compare", ext=".txt", basedir=report_dir)
        )

        log.info(f"Saving compare json to {json_output_file}")
        log.info(f"Saving compare table to {txt_output_file}")

        dump_json_file(json_output_file, comparison_dict, indent=True)
        with open(txt_output_file, "w") as f:
            f.write(f"High-level Comparison:\n{high_level_table}\n")
            f.write(f"Mid-level Comparison:\n{mid_level_table}\n")
            f.write(f"Low-level Comparison:\n{low_level_table}\n")
