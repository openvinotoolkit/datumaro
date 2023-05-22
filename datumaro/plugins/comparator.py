# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from unittest import TestCase

from attr import attrib, attrs
from texttable import Texttable

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset import Dataset
from datumaro.components.operations import (
    compute_ann_statistics,
    compute_image_statistics,
    match_classes,
    match_items_by_id,
    match_items_by_image_hash,
)
from datumaro.plugins.shift_analyzer import ShiftAnalyzer
from datumaro.util import filter_dict, find
from datumaro.util.attrs_util import default_if_none
import os.path as osp
from datumaro.cli.util.project import generate_next_file_name
import logging as log
from datumaro.util import dump_json_file
import os

@attrs
class Comparator:
    # match_images: bool = attrib(kw_only=True, default=False)
    # ignored_fields = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    # ignored_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))
    # ignored_item_attrs = attrib(kw_only=True, factory=set, validator=default_if_none(set))

    # _test: TestCase = attrib(init=False)
    # errors: list = attrib(init=False)

    # def __attrs_post_init__(self):
    #     self._test = TestCase()
    #     self._test.maxDiff = None

    # def _match_items(self, src, tgt):
    #     if self.match_images:
    #         return match_items_by_image_hash(src, tgt)
    #     else:
    #         return match_items_by_id(src, tgt)

    # def _compare_categories(self, src, tgt):
    #     test = self._test
    #     errors = self.errors

    #     try:
    #         test.assertEqual(sorted(src, key=lambda t: t.value), sorted(tgt, key=lambda t: t.value))
    #     except AssertionError as e:
    #         errors.append({"type": "categories", "message": str(e)})

    #     if AnnotationType.label in src:
    #         try:
    #             test.assertEqual(
    #                 src[AnnotationType.label].items,
    #                 tgt[AnnotationType.label].items,
    #             )
    #         except AssertionError as e:
    #             errors.append({"type": "labels", "message": str(e)})
    #     if AnnotationType.mask in src:
    #         try:
    #             test.assertEqual(
    #                 src[AnnotationType.mask].colormap,
    #                 tgt[AnnotationType.mask].colormap,
    #             )
    #         except AssertionError as e:
    #             errors.append({"type": "colormap", "message": str(e)})
    #     if AnnotationType.points in src:
    #         try:
    #             test.assertEqual(
    #                 src[AnnotationType.points].items,
    #                 tgt[AnnotationType.points].items,
    #             )
    #         except AssertionError as e:
    #             errors.append({"type": "points", "message": str(e)})

    # def _compare_annotations(self, src, tgt):
    #     ignored_fields = self.ignored_fields
    #     ignored_attrs = self.ignored_attrs

    #     a_fields = {k: None for k in src.as_dict() if k in ignored_fields}
    #     b_fields = {k: None for k in tgt.as_dict() if k in ignored_fields}
    #     if "attributes" not in ignored_fields:
    #         a_fields["attributes"] = filter_dict(src.attributes, ignored_attrs)
    #         b_fields["attributes"] = filter_dict(tgt.attributes, ignored_attrs)

    #     result = src.wrap(**a_fields) == tgt.wrap(**b_fields)

    #     return result

    # def _compare_items(self, item_a, item_b):
    #     test = self._test

    #     a_id = (item_a.id, item_a.subset)
    #     b_id = (item_b.id, item_b.subset)

    #     matched = []
    #     unmatched = []
    #     errors = []

    #     try:
    #         test.assertEqual(
    #             filter_dict(item_a.attributes, self.ignored_item_attrs),
    #             filter_dict(item_b.attributes, self.ignored_item_attrs),
    #         )
    #     except AssertionError as e:
    #         errors.append({"type": "item_attr", "a_item": a_id, "b_item": b_id, "message": str(e)})

    #     b_annotations = item_b.annotations[:]
    #     for ann_a in item_a.annotations:
    #         ann_b_candidates = [x for x in item_b.annotations if x.type == ann_a.type]

    #         ann_b = find(
    #             enumerate(self._compare_annotations(ann_a, x) for x in ann_b_candidates),
    #             lambda x: x[1],
    #         )
    #         if ann_b is None:
    #             unmatched.append(
    #                 {
    #                     "item": a_id,
    #                     "source": "src",
    #                     "ann": str(ann_a),
    #                 }
    #             )
    #             continue
    #         else:
    #             ann_b = ann_b_candidates[ann_b[0]]

    #         b_annotations.remove(ann_b)  # avoid repeats
    #         matched.append({"a_item": a_id, "b_item": b_id, "src": str(ann_a), "tgt": str(ann_b)})

    #     for ann_b in b_annotations:
    #         unmatched.append({"item": b_id, "source": "tgt", "ann": str(ann_b)})

    #     return matched, unmatched, errors

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
    def save_compare_report(high_level_table, mid_level_table, low_level_table, comparison_dict, path: str) -> None:

        os.makedirs(path, exist_ok=True)
        json_output_file = osp.join(
            path, generate_next_file_name("compare", ext=".json", basedir=path)
        )
        txt_output_file = osp.join(
            path, generate_next_file_name("compare", ext=".txt", basedir=path)
        )

        log.info("Saving compare json to '%s'" % json_output_file)
        log.info("Saving compare table to '%s'" % txt_output_file)

        dump_json_file(json_output_file, comparison_dict, indent=True)
        with open(txt_output_file, "w") as f:
            f.write(f"High-level Comparison:\n{high_level_table}\n")
            f.write(f"Mid-level Comparison:\n{mid_level_table}\n")
            f.write(f"Low-level Comparison:\n{low_level_table}\n")

        
