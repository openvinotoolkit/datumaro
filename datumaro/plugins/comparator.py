# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp

from attr import attrs
from texttable import Texttable

from datumaro.cli.util.project import generate_next_file_name
from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.operations import compute_ann_statistics, compute_image_statistics
from datumaro.plugins.shift_analyzer import ShiftAnalyzer
from datumaro.util import dump_json_file


@attrs
class Comparator:
    @staticmethod
    def _analyze_dataset(dataset):
        label_cat = dataset.categories().get(AnnotationType.label, LabelCategories())
        labels = set(c.name for c in label_cat)

        dataset_format = dataset.format
        dataset_classes = labels
        image_stats = compute_image_statistics(dataset)
        ann_stats = compute_ann_statistics(dataset)

        return dataset_format, dataset_classes, image_stats, ann_stats

    @staticmethod
    def generate_high_level_comparison_table(src_info, tgt_info):
        src_dataset_format, src_dataset_classes, src_image_stats, src_ann_stats = src_info
        tgt_dataset_format, tgt_dataset_classes, tgt_image_stats, tgt_ann_stats = tgt_info
        table = Texttable()

        table.set_cols_align(["l", "l", "l"])
        table.set_cols_valign(["m", "m", "m"])

        rows = [
            ["Field", "Source", "Target"],
            ["Format", src_dataset_format, tgt_dataset_format],
            ["Number of classes", len(src_dataset_classes), len(tgt_dataset_classes)],
            [
                "Intersect classes",
                ", ".join(sorted(list(src_dataset_classes.intersection(tgt_dataset_classes)))),
                ",".join(sorted(list(src_dataset_classes.intersection(tgt_dataset_classes)))),
            ],
            [
                "Classes",
                ", ".join(sorted(src_dataset_classes)),
                ", ".join(sorted(tgt_dataset_classes)),
            ],
            [
                "Images count",
                src_image_stats["dataset"]["images count"],
                tgt_image_stats["dataset"]["images count"],
            ],
            [
                "Unique images count",
                src_image_stats["dataset"]["unique images count"],
                tgt_image_stats["dataset"]["unique images count"],
            ],
            [
                "Repeated images count",
                src_image_stats["dataset"]["repeated images count"],
                tgt_image_stats["dataset"]["repeated images count"],
            ],
            [
                "Annotations count",
                src_ann_stats["annotations count"],
                tgt_ann_stats["annotations count"],
            ],
            [
                "Unannotated images count",
                src_ann_stats["unannotated images count"],
                tgt_ann_stats["unannotated images count"],
            ],
        ]

        data_dict = {row[0]: row[1:] for row in rows[1:]}
        table.add_rows(rows)
        return table.draw(), data_dict

    @staticmethod
    def generate_mid_level_comparison_table(src_info, tgt_info):
        _, _, src_image_stats, src_ann_stats = src_info
        _, _, tgt_image_stats, tgt_ann_stats = tgt_info

        table = Texttable()

        table.set_cols_align(["l", "l", "l"])
        table.set_cols_valign(["m", "m", "m"])

        rows = [
            ["Field", "Source", "Target"],
        ]

        # sort by subset names
        subset_names = sorted(
            set(src_image_stats["subsets"].keys()).union(tgt_image_stats["subsets"].keys())
        )

        for subset_name in subset_names:
            src_subset_data = src_image_stats["subsets"].get(subset_name, {})
            tgt_subset_data = tgt_image_stats["subsets"].get(subset_name, {})
            mean_str_src = (
                ", ".join(f"{val:6.2f}" for val in src_subset_data.get("image mean", []))
                if "image mean" in src_subset_data
                else ""
            )
            std_str_src = (
                ", ".join(f"{val:6.2f}" for val in src_subset_data.get("image std", []))
                if "image std" in src_subset_data
                else ""
            )
            mean_str_tgt = (
                ", ".join(f"{val:6.2f}" for val in tgt_subset_data.get("image mean", []))
                if "image mean" in tgt_subset_data
                else ""
            )
            std_str_tgt = (
                ", ".join(f"{val:6.2f}" for val in tgt_subset_data.get("image std", []))
                if "image std" in tgt_subset_data
                else ""
            )
            rows.append([f"{subset_name} - Image Mean", mean_str_src, mean_str_tgt])
            rows.append([f"{subset_name} - Image Std", std_str_src, std_str_tgt])

        label_names = sorted(
            set(src_ann_stats["annotations"]["labels"]["distribution"].keys()).union(
                tgt_ann_stats["annotations"]["labels"]["distribution"].keys()
            )
        )

        for label_name in label_names:
            count_dist_src = src_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_dist_tgt = tgt_ann_stats["annotations"]["labels"]["distribution"].get(
                label_name, [0, 0.0]
            )
            count_src, dist_src = count_dist_src if count_dist_src[0] != 0 else ["", ""]
            count_tgt, dist_tgt = count_dist_tgt if count_dist_tgt[0] != 0 else ["", ""]
            rows.append(
                [
                    f"Label - {label_name}",
                    f"imgs: {count_src}, percent: {dist_src:.4f}" if count_src != "" else "",
                    f"imgs: {count_tgt}, percent: {dist_tgt:.4f}" if count_tgt != "" else "",
                ]
            )

        table.add_rows(rows)
        data_dict = {row[0]: row[1:] for row in rows[1:]}
        return table.draw(), data_dict

    @staticmethod
    def generate_low_level_comparison_table(src_dataset, tgt_dataset):
        shift = ShiftAnalyzer()
        cov_shift = shift.compute_covariate_shift([src_dataset, tgt_dataset])
        label_shift = shift.compute_label_shift([src_dataset, tgt_dataset])

        table = Texttable()
        table.set_cols_align(["l", "l"])
        table.set_cols_valign(["m", "m"])

        rows = [
            ["Field", "Value"],
            ["Covariate shift", cov_shift],
            ["Label shift", label_shift],
        ]

        table.add_rows(rows)
        data_dict = {row[0]: row[1:] for row in rows[1:]}
        return table.draw(), data_dict

    def compare_datasets(self, src, tgt):
        src_info = self._analyze_dataset(src)
        tgt_info = self._analyze_dataset(tgt)

        high_level_table, high_level_dict = self.generate_high_level_comparison_table(
            src_info, tgt_info
        )
        mid_level_table, mid_level_dict = self.generate_mid_level_comparison_table(
            src_info, tgt_info
        )
        low_level_table, low_level_dict = self.generate_low_level_comparison_table(src, tgt)

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
