import logging as log
import os
import os.path as osp
from textwrap import wrap
from typing import Any, Dict, List, Set, Tuple

from attr import attrs
from tabulate import tabulate

from datumaro.cli.util.project import generate_next_file_name
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.plugins.shift_analyzer import ShiftAnalyzer
from datumaro.util import dump_json_file


@attrs
class Comparator:
    """
    Comparator is a class used for comparing datasets and generating comparison reports.
    """

    @staticmethod
    def _extract_labels(dataset: Any) -> Set[str]:
        """Extracts labels from the dataset.

        Args:
            dataset: An instance of a dataset class.

        Returns:
            A set of labels present in the dataset.
        """
        label_cat = dataset.categories().get(AnnotationType.label, LabelCategories())
        return set(c.name for c in label_cat)

    @staticmethod
    def _compute_statistics(dataset: Any) -> Tuple[Dict, Dict]:
        """Computes image and annotation statistics of the dataset.

        Args:
            dataset: An instance of a dataset class.

        Returns:
            A tuple containing image statistics and annotation statistics.
        """
        image_stats = compute_image_statistics(dataset)
        ann_stats = compute_ann_statistics(dataset)
        return image_stats, ann_stats

    def _analyze_dataset(self, dataset: Any) -> Tuple[str, Set[str], Dict, Dict]:
        """Analyzes the dataset to get labels, format, and statistics.

        Args:
            dataset: An instance of a dataset class.

        Returns:
            A tuple containing dataset format, set of label names, image statistics,
            and annotation statistics.
        """
        dataset_format = dataset.format
        dataset_labels = self._extract_labels(dataset)
        image_stats, ann_stats = self._compute_statistics(dataset)
        return dataset_format, dataset_labels, image_stats, ann_stats

    @staticmethod
    def _create_table(headers: List[str], rows: List[List[str]]) -> str:
        """Creates a table with the given headers and rows using the tabulate module.

        Args:
            headers: A list containing table headers.
            rows: A list containing table rows.

        Returns:
            A string representation of the table.
        """

        def wrapfunc(item):
            """Wrap a item consisted of text, returning a list of wrapped lines."""
            max_len = 35
            return "\n".join(wrap(item, max_len))

        wrapped_rows = []
        for row in rows:
            new_row = [wrapfunc(item) for item in row]
            wrapped_rows.append(new_row)

        return tabulate(wrapped_rows, headers, tablefmt="grid")

    @staticmethod
    def _create_dict(rows: List[List[str]]) -> Dict[str, List[str]]:
        """Creates a dictionary from the rows of the table.

        Args:
            rows: A list containing table rows.

        Returns:
            A dictionary where the key is the first element of a row and the value is
            the rest of the row.
        """
        data_dict = {row[0]: row[1:] for row in rows[1:]}
        return data_dict

    def _create_high_level_comparison_table(
        self, first_info: Tuple, second_info: Tuple
    ) -> Tuple[str, Dict]:
        """Generates a high-level comparison table.

        Args:
            first_info: A tuple containing information about the first dataset.
            second_info: A tuple containing information about the second dataset.

        Returns:
            A tuple containing the table as a string and a dictionary representing the data
            of the table.
        """
        first_format, first_labels, first_image_stats, first_ann_stats = first_info
        second_format, second_labels, second_image_stats, second_ann_stats = second_info

        headers = ["Field", "First", "Second"]

        rows = [
            ["Format", first_format, second_format],
            ["Number of classes", str(len(first_labels)), str(len(second_labels))],
            [
                "Common classes",
                ", ".join(sorted(list(first_labels.intersection(second_labels)))),
                ", ".join(sorted(list(second_labels.intersection(first_labels)))),
            ],
            ["Classes", ", ".join(sorted(first_labels)), ", ".join(sorted(second_labels))],
            [
                "Images count",
                str(first_image_stats["dataset"]["images count"]),
                str(second_image_stats["dataset"]["images count"]),
            ],
            [
                "Unique images count",
                str(first_image_stats["dataset"]["unique images count"]),
                str(second_image_stats["dataset"]["unique images count"]),
            ],
            [
                "Repeated images count",
                str(first_image_stats["dataset"]["repeated images count"]),
                str(second_image_stats["dataset"]["repeated images count"]),
            ],
            [
                "Annotations count",
                str(first_ann_stats["annotations count"]),
                str(second_ann_stats["annotations count"]),
            ],
            [
                "Unannotated images count",
                str(first_ann_stats["unannotated images count"]),
                str(second_ann_stats["unannotated images count"]),
            ],
        ]

        table = self._create_table(headers, rows)
        data_dict = self._create_dict(rows)

        return table, data_dict

    def _create_mid_level_comparison_table(
        self, first_info: Tuple, second_info: Tuple
    ) -> Tuple[str, Dict]:
        """Generates a mid-level comparison table.

        Args:
            first_info: A tuple containing information about the first dataset.
            second_info: A tuple containing information about the second dataset.

        Returns:
            A tuple containing the table as a string and a dictionary representing the data
            of the table.
        """
        _, _, first_image_stats, first_ann_stats = first_info
        _, _, second_image_stats, second_ann_stats = second_info

        headers = ["Field", "First", "Second"]

        rows = []

        first_subsets = sorted(list(first_image_stats["subsets"].keys()))
        second_subsets = sorted(list(second_image_stats["subsets"].keys()))

        subset_names = first_subsets.copy()
        subset_names.extend(item for item in second_subsets if item not in first_subsets)

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

        first_labels = sorted(list(first_ann_stats["annotations"]["labels"]["distribution"].keys()))
        second_labels = sorted(
            list(second_ann_stats["annotations"]["labels"]["distribution"].keys())
        )

        label_names = first_labels.copy()
        label_names.extend(item for item in second_labels if item not in first_labels)

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

        table = self._create_table(headers, rows)
        data_dict = self._create_dict(rows)

        return table, data_dict

    def _create_low_level_comparison_table(
        self, first_dataset: Any, second_dataset: Any
    ) -> Tuple[str, Dict]:
        """Generates a low-level comparison table.

        Args:
            first_dataset: The first dataset to compare.
            second_dataset: The second dataset to compare.

        Returns:
            A tuple containing the table as a string and a dictionary representing the data
            of the table.
        """
        shift_analyzer = ShiftAnalyzer()
        cov_shift = shift_analyzer.compute_covariate_shift([first_dataset, second_dataset])
        label_shift = shift_analyzer.compute_label_shift([first_dataset, second_dataset])

        headers = ["Field", "Value"]

        rows = [
            ["Covariate shift", str(cov_shift)],
            ["Label shift", str(label_shift)],
        ]

        table = self._create_table(headers, rows)
        data_dict = self._create_dict(rows)

        return table, data_dict

    def compare_datasets(self, first: Any, second: Any) -> Tuple[str, str, str, Dict]:
        """Compares two datasets and generates comparison reports.

        Args:
            first: The first dataset to compare.
            second: The second dataset to compare.

        Returns:
            A tuple containing high-level table, mid-level table, low-level table, and a
            dictionary representation of the comparison.
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
        high_level_table: str,
        mid_level_table: str,
        low_level_table: str,
        comparison_dict: Dict,
        report_dir: str,
    ) -> None:
        """Saves the comparison report to JSON and text files.

        Args:
            high_level_table: High-level comparison table as a string.
            mid_level_table: Mid-level comparison table as a string.
            low_level_table: Low-level comparison table as a string.
            comparison_dict: A dictionary containing the comparison data.
            report_dir: A string representing the directory to save the report files.
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
