# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from collections import defaultdict
from unittest import TestCase

import numpy as np
from streamlit.testing.v1 import AppTest

from tests.requirements import Requirements, mark_requirement

cwd = os.getcwd()
import sys

sys.path.append(os.path.join(cwd, "gui"))

from gui.datumaro_gui.components.single.tabs.analyze import (
    CHART_KWARGS,
    get_anomaly_info,
    get_attr_dist,
    get_dataset_info,
    get_grid_data_val_invalid_value,
    get_grid_data_val_missing_annotations,
    get_grid_data_val_multiple_labels,
    get_grid_data_val_negative_length,
    get_image_size_dist,
    get_label_dist,
    get_num_anns_by_type,
    get_num_images_by_subset,
    get_radar_data_for_label_dist_by_type,
    get_repeated_images,
    get_segments_dist,
    get_tab_data_for_label_dist_by_type,
    get_tab_data_for_val_attr_dist,
    get_tab_data_for_val_label_dist,
    get_tab_data_point_dist_in_label,
    get_unannotated_images,
    get_validation_summary,
)
from gui.datumaro_gui.utils.drawing import Dashboard


def run_analyze():
    import os

    import streamlit as state
    from streamlit import session_state as state

    from gui.datumaro_gui.components.single import tabs
    from gui.datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
    from gui.datumaro_gui.utils.dataset.state import reset_state, single_state_keys
    from gui.datumaro_gui.utils.page import init_func

    from tests.utils.assets import get_test_asset_path

    init_func(state.get("IMAGE_BACKEND", None))
    reset_state(single_state_keys, state)

    dataset_dir = get_test_asset_path("datumaro_dataset")
    data_helper = SingleDatasetHelper(dataset_dir)
    state["data_helper"] = data_helper
    uploaded_file = os.path.basename(dataset_dir)
    state["uploaded_file"] = uploaded_file

    data_helper = state["data_helper"]

    data_helper.import_dataset("datumaro")

    tabs.call_analyze()


class AnalyzeTest(TestCase):
    @property
    def stats_image(self):
        return {
            "dataset": {
                "images count": 1,
                "unique images count": 1,
                "repeated images count": 0,
                "repeated images": [],
            },
            "subsets": {
                "train": {
                    "images count": 1,
                    "image mean": [0.0, 0.0, 0.0],
                    "image std": [0.0, 0.0, 0.0],
                }
            },
        }

    @property
    def stats_anns(self):
        return {
            "images count": 1,
            "annotations count": 2,
            "unannotated images count": 0,
            "unannotated images": [],
            "annotations by type": {
                "unknown": {"count": 0},
                "label": {"count": 2},
                "mask": {"count": 0},
                "points": {"count": 0},
                "polygon": {"count": 0},
                "polyline": {"count": 0},
                "bbox": {"count": 0},
                "caption": {"count": 0},
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "ellipse": {"count": 0},
                "hash_key": {"count": 0},
                "feature_vector": {"count": 0},
                "tabular": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 2,
                    "distribution": defaultdict(lambda: [0, 0]),
                    "attributes": {},
                },
                "segments": {
                    "avg. area": 0.0,
                    "area distribution": [],
                    "pixel distribution": defaultdict(lambda: [0, 0]),
                },
            },
        }

    @property
    def image_size_info(self):
        return {
            "by_subsets": defaultdict(
                list, {"test": [], "train": [{"x": 6, "y": 8}], "validation": []}
            ),
            "by_labels": defaultdict(
                list, {"bicycle": [{"x": 6, "y": 8}], "mary": [], "car": [], "tom": []}
            ),
            "image_size": {"mean": np.array([6.0, 8.0]), "std": np.array([0.0, 0.0])},
        }

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_analyze_page_open(self):
        """Test if the page of analyze tab is opened correctly."""
        at = AppTest.from_function(run_analyze, default_timeout=600).run()

        assert at.session_state.cls_anomaly_info
        assert at.session_state.cls_summary
        assert at.session_state.defined_label

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_dataset_info(self):
        """Test if get_dataset_info function of analyze component is worked out correctly."""
        stats_image = self.stats_image
        stats_anns = self.stats_anns
        image_mean = np.array([0.0, 0.0])
        n_labels = 2
        dataset_info = get_dataset_info(stats_image, stats_anns, image_mean, n_labels)

        assert dataset_info["n_images"] == stats_image["dataset"]["images count"]
        assert dataset_info["n_unique"] == stats_image["dataset"]["unique images count"]
        assert dataset_info["n_repeated"] == stats_image["dataset"]["repeated images count"]
        assert dataset_info["avg_w"] == image_mean[1]
        assert dataset_info["avg_h"] == image_mean[0]
        assert dataset_info["n_subsets"] == len(stats_image["subsets"])
        assert dataset_info["n_anns"] == stats_anns["annotations count"]
        assert dataset_info["n_unannotated"] == stats_anns["unannotated images count"]
        assert dataset_info["n_labels"] == n_labels

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_dataset_info_no_subsets(self):
        """Test if get_dataset_info function with no subsets of analyze component is worked out correctly."""
        stats_image = self.stats_image
        stats_anns = self.stats_anns
        image_mean = np.array([0.0, 0.0])
        n_labels = 2

        stats_image["subsets"] = []
        dataset_info = get_dataset_info(stats_image, stats_anns, image_mean, n_labels)

        assert dataset_info["n_subsets"] == 0

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_dataset_info_no_annotations(self):
        """Test if get_dataset_info function with no annotations of analyze component is worked out correctly."""
        stats_image = self.stats_image
        stats_anns = self.stats_anns
        image_mean = np.array([0.0, 0.0])
        n_labels = 2

        stats_anns["annotations count"] = 0
        dataset_info = get_dataset_info(stats_image, stats_anns, image_mean, n_labels)

        assert dataset_info["n_anns"] == 0
        assert dataset_info["n_unannotated"] == stats_anns["unannotated images count"]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_num_images_by_subset(self):
        """Test if get_num_images_by_subset function of analyze component is worked out correctly."""
        stats_image = {
            "subsets": {
                "subset1": {"images count": 10},
                "subset2": {"images count": 20},
            }
        }
        expected_output = [
            {"id": "subset1", "label": "subset1", "value": 10},
            {"id": "subset2", "label": "subset2", "value": 20},
        ]
        result = get_num_images_by_subset(stats_image)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_num_images_by_subset_empty_subsets(self):
        """Test if get_num_images_by_subset function with empty subsets of analyze component is worked out correctly."""
        stats_image = self.stats_image
        stats_image["subsets"] = {}
        result = get_num_images_by_subset(stats_image)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_image_size_dist(self):
        """Test if get_image_size_dist function of analyze component is worked out correctly."""
        image_size_info = {
            "by_subsets": {"subset1": [(100, 200), (300, 400)]},
            "by_labels": {"label1": [(50, 50)]},
        }
        expected_tab_data = [
            {
                "title": "By Subsets (2 Images)",
                "data": [{"id": "subset1", "data": [(100, 200), (300, 400)]}],
                "chart_type": Dashboard.Chart.ScatterPlot,
                "chart_kwargs": CHART_KWARGS,  # Include expected chart_kwargs
            },
            {
                "title": "By Labels (1 Labels)",
                "data": [{"id": "label1", "data": [(50, 50)]}],
                "chart_type": Dashboard.Chart.ScatterPlot,
                "chart_kwargs": CHART_KWARGS,  # Include expected chart_kwargs
            },
        ]
        result = get_image_size_dist(image_size_info)
        assert result == expected_tab_data

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_image_size_dist_empty_data(self):
        """Test if get_image_size_dist function with empty data of analyze component is worked out correctly."""
        image_size_info = {"by_subsets": {}, "by_labels": {}}
        result = get_image_size_dist(image_size_info)
        assert result is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_repeated_images_basic(self):
        """Test if get_repeated_images function of analyze component is worked out correctly."""
        stats_image = {
            "dataset": {
                "repeated images": [
                    [(1, "subset1"), (2, "subset2")],
                    [(3, "subset1")],
                ]
            }
        }
        expected_grid_data = [
            {"id": 0, "repeated": "subset1-1, subset2-2"},
            {"id": 1, "repeated": "subset1-3"},
        ]
        result = get_repeated_images(stats_image)
        assert result == expected_grid_data

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_repeated_images_empty_data(self):
        """Test if get_repeated_images function with empty data of analyze component is worked out correctly."""
        stats_image = {"dataset": {"repeated images": []}}
        result = get_repeated_images(stats_image)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_unannotated_images(self):
        """Test if get_unannotated_images function of analyze component is worked out correctly."""
        stats_ann = {"unannotated images": ["image1", "image2"]}
        expected_grid_data = [{"id": 0, "item_id": "image1"}, {"id": 1, "item_id": "image2"}]
        result = get_unannotated_images(stats_ann)
        assert result == expected_grid_data

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_unannotated_images_empty_data(self):
        """Test if get_unannotated_images function with empty data of analyze component is worked out correctly."""
        stats_ann = {"unannotated images": []}
        result = get_unannotated_images(stats_ann)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_num_anns_by_type(self):
        """Test if get_num_anns_by_type function of analyze component is worked out correctly."""
        stats_ann = {
            "annotations by type": {
                "label": {"count": 10},
                "bbox": {"count": 5},
                "polygon": {"count": 0},  # Count of 0 should be excluded
            }
        }
        expected_output = [
            {"id": "label", "label": "label", "value": 10},
            {"id": "bbox", "label": "bbox", "value": 5},
        ]
        result = get_num_anns_by_type(stats_ann)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_num_anns_by_type_empty_data(self):
        """Test if get_num_anns_by_type function with empty data of analyze component is worked out correctly."""
        stats_ann = {"annotations by type": {}}
        result = get_num_anns_by_type(stats_ann)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_label_dist(self):
        """Test if get_label_dist function of analyze component is worked out correctly."""
        stats_ann = {
            "annotations": {
                "labels": {
                    "distribution": {
                        "label1": [10],
                        "label2": [5],
                    }
                }
            }
        }
        expected_output = [
            {"id": "label1", "label": "label1", "value": 10},
            {"id": "label2", "label": "label2", "value": 5},
        ]
        result = get_label_dist(stats_ann)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_label_dist_empty_data(self):
        """Test if get_label_dist function with empty data of analyze component is worked out correctly."""
        stats_ann = {"annotations": {"labels": {"distribution": {}}}}
        result = get_label_dist(stats_ann)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_attr_dist(self):
        """Test if get_attr_dist function of analyze component is worked out correctly."""
        mock_attributes = {
            "attr1": {
                "values present": ["val1", "val2"],
                "distribution": {"val1": [5], "val2": [3]},
            },
            "attr2": {"values present": ["val3"], "distribution": {"val3": [2]}},
        }
        stats_ann = {"annotations": {"labels": {"attributes": mock_attributes}}}
        expected_output = [
            {
                "title": "attr1",
                "data": [
                    {"id": "val1", "label": "val1", "value": 5},
                    {"id": "val2", "label": "val2", "value": 3},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
            {
                "title": "attr2",
                "data": [{"id": "val3", "label": "val3", "value": 2}],
                "chart_type": Dashboard.Chart.Pie,
            },
        ]
        result = get_attr_dist(stats_ann)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_attr_dist_empty_data(self):
        """Test if get_attr_dist function with empty data of analyze component is worked out correctly."""
        stats_ann = {"annotations": {"labels": {"attributes": {}}}}
        result = get_attr_dist(stats_ann)
        assert result is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_segments_dist(self):
        """Test if get_segments_dist function of analyze component is worked out correctly."""
        mock_segments = {
            "area distribution": [
                {"min": 10, "max": 20, "count": 5},
                {"min": 30, "max": 40, "count": 3},
            ],
            "pixel distribution": {"class1": [10], "class2": [5]},
        }
        stats_ann = {"annotations": {"segments": mock_segments}}
        expected_output = [
            {
                "title": "Bbox Area Distribution",
                "data": [
                    {"id": "10.00~20.00", "label": "10.00~20.00", "value": 5},
                    {"id": "30.00~40.00", "label": "30.00~40.00", "value": 3},
                ],
                "chart_type": Dashboard.Chart.Bar,
            },
            {
                "title": "Pixel Distribution",
                "data": [
                    {"id": "class1", "label": "class1", "value": 10},
                    {"id": "class2", "label": "class2", "value": 5},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
        ]
        result = get_segments_dist(stats_ann)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_tab_data_for_val_label_dist(self):
        """Test if get_tab_data_for_val_label_dist function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 5, "label2": 3},
                    "undefined_labels": {"label3": 2},
                }
            }
        }
        expected_output = [
            {
                "title": "Defined Labels (8)",
                "data": [
                    {"id": "label1", "label": "label1", "value": 5},
                    {"id": "label2", "label": "label2", "value": 3},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
            {
                "title": "Undefined Labels(2)",
                "data": [{"id": "label3", "label": "label3", "value": 2}],
                "chart_type": Dashboard.Chart.Pie,
            },
        ]
        result = get_tab_data_for_val_label_dist(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_tab_data_for_val_attr_dist(self):
        """Test if get_tab_data_for_val_attr_dist function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "attribute_distribution": {
                    "defined_attributes": {
                        "label1": {"attr1": {"distribution": {"val1": 5, "val2": 3}}},
                        "label2": {"attr2": {"distribution": {"val3": 2}}},
                    },
                    "undefined_attributes": {"label3": {"attr3": {"distribution": {"val4": 1}}}},
                }
            }
        }
        expected_output = [
            {
                "title": "Defined Attributes (10)",
                "data": {
                    "id": "nivo",
                    "children": [
                        {
                            "id": "label1/attr1",
                            "children": [
                                {"id": "label1/attr1-val1", "value": 5},
                                {"id": "label1/attr1-val2", "value": 3},
                            ],
                        },
                        {
                            "id": "label2/attr2",
                            "children": [{"id": "label2/attr2-val3", "value": 2}],
                        },
                    ],
                },
                "chart_type": Dashboard.Chart.Sunburst,
                "chart_kwargs": {
                    "enableArcLabels": True,
                    "childColor": {"from": "color", "modifiers": [["brighter", "0.5"]]},
                },
            },
            {
                "title": "Undefined Attributes (1)",
                "data": {
                    "id": "nivo",
                    "children": [
                        {
                            "id": "label3/attr3",
                            "children": [{"id": "label3/attr3-val4", "value": 1}],
                        }
                    ],
                },
                "chart_type": Dashboard.Chart.Sunburst,
            },
        ]
        result = get_tab_data_for_val_attr_dist(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_missing_annotations(self):
        """Test if get_grid_data_val_missing_annotations function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "items_missing_annotation": [
                    ("item1", "train"),
                    ("item2", "validation"),
                    ("item3", "test"),
                ]
            }
        }
        expected_output = [
            {"id": 0, "item_id": "item1", "subset": "train"},
            {"id": 1, "item_id": "item2", "subset": "validation"},
            {"id": 2, "item_id": "item3", "subset": "test"},
        ]
        result = get_grid_data_val_missing_annotations(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_missing_annotations_empty_data(self):
        """Test if get_grid_data_val_missing_annotations function with empty data of analyze component is worked out correctly."""
        mock_report = {"statistics": {"items_missing_annotation": []}}
        result = get_grid_data_val_missing_annotations(mock_report)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_negative_length(self):
        """Test if get_grid_data_val_negative_length function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "items_with_negative_length": {
                    ("item1", "train"): {"ann1": [-1.5, -2.0], "ann2": [-3.0]},
                    ("item2", "validation"): {"ann3": [-0.5]},
                }
            }
        }
        expected_output = [
            {
                "id": 0,
                "subset": "train",
                "item_id": "item1",
                "ann_id": "ann1",
                "values": [-1.5, -2.0],
            },
            {"id": 1, "subset": "train", "item_id": "item1", "ann_id": "ann2", "values": [-3.0]},
            {
                "id": 2,
                "subset": "validation",
                "item_id": "item2",
                "ann_id": "ann3",
                "values": [-0.5],
            },
        ]
        result = get_grid_data_val_negative_length(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_negative_length_empty_data(self):
        """Test if get_grid_data_val_negative_length function with empty data of analyze component is worked out correctly."""
        mock_report = {"statistics": {"items_with_negative_length": {}}}
        result = get_grid_data_val_negative_length(mock_report)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_invalid_value(self):
        """Test if get_grid_data_val_invalid_value function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "items_with_invalid_value": {
                    ("item1", "train"): {"ann1": ["NaN", "inf"], "ann2": ["-inf"]},
                    ("item2", "test"): {"ann3": ["N/A"]},
                }
            }
        }
        expected_output = [
            {
                "id": 0,
                "subset": "train",
                "item_id": "item1",
                "ann_id": "ann1",
                "values": ["NaN", "inf"],
            },
            {"id": 1, "subset": "train", "item_id": "item1", "ann_id": "ann2", "values": ["-inf"]},
            {"id": 2, "subset": "test", "item_id": "item2", "ann_id": "ann3", "values": ["N/A"]},
        ]
        result = get_grid_data_val_invalid_value(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_invalid_value_empty_data(self):
        """Test if get_grid_data_val_invalid_value function with empty data of analyze component is worked out correctly."""
        mock_report = {"statistics": {"items_with_invalid_value": {}}}
        result = get_grid_data_val_invalid_value(mock_report)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_multiple_labels(self):
        """Test if get_grid_data_val_multiple_labels function of analyze component is worked out correctly."""
        mock_report = {
            "statistics": {
                "items_with_multiple_labels": [
                    ("item1", "train"),
                    ("item2", "validation"),
                    ("item3", "test"),
                ]
            }
        }
        expected_output = [
            {"id": 0, "item_id": "item1", "subset": "train"},
            {"id": 1, "item_id": "item2", "subset": "validation"},
            {"id": 2, "item_id": "item3", "subset": "test"},
        ]
        result = get_grid_data_val_multiple_labels(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_grid_data_val_multiple_labels_empty_data(self):
        """Test if get_grid_data_val_multiple_labels function with empty data of analyze component is worked out correctly."""
        mock_report = {"statistics": {"items_with_multiple_labels": []}}
        result = get_grid_data_val_multiple_labels(mock_report)
        assert result == []

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_validation_summary(self):
        """Test if get_validation_summary function of analyze component is worked out correctly."""
        mock_report = {
            "validation_reports": [
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeB"},
                {"anomaly_type": "TypeC"},
            ]
        }
        expected_output = [
            {"id": "TypeA", "label": "TypeA", "value": 2},
            {"id": "TypeB", "label": "TypeB", "value": 1},
            {"id": "TypeC", "label": "TypeC", "value": 1},
        ]
        result = get_validation_summary(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_validation_summary_empty_data(self):
        """Test if get_validation_summary function with empty data of analyze component is worked out correctly."""
        mock_report = {"validation_reports": []}
        result = get_validation_summary(mock_report)
        assert result is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_anomaly_info(self):
        """Test if get_anomaly_info function of analyze component is worked out correctly."""
        mock_report = {
            "validation_reports": [
                {
                    "anomaly_type": "TypeA",
                    "subset": "train",
                    "item_id": "123",
                    "description": "Missing annotation",
                },
                {"anomaly_type": "TypeB", "description": "Invalid value"},
            ]
        }
        expected_output = [
            {
                "id": 0,
                "anomaly": "TypeA",
                "subset": "train",
                "item_id": "123",
                "description": "Missing annotation",
            },
            {
                "id": 1,
                "anomaly": "TypeB",
                "subset": "None",
                "item_id": "None",
                "description": "Invalid value",
            },
        ]
        result = get_anomaly_info(mock_report)
        assert result == expected_output

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_anomaly_info_empty_data(self):
        """Test if get_anomaly_info function with empty data of analyze component is worked out correctly."""
        mock_report = {"validation_reports": []}
        result = get_anomaly_info(mock_report)
        assert result is None

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_tab_data_for_label_dist_by_type(self):
        """Test if get_tab_data_for_label_dist_by_type function of analyze component is worked out correctly."""
        val_cls = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 5, "label2": 3},
                    "undefined_labels": {"label3": 2},
                }
            },
            "validation_reports": [
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeB"},
                {"anomaly_type": "TypeC"},
            ],
            "summary": {"errors": 4, "warnings": 1, "infos": 1},
        }
        val_det = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 0, "label2": 0},
                    "undefined_labels": {},
                }
            },
            "validation_reports": [
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeB"},
                {"anomaly_type": "TypeC"},
            ],
            "summary": {"errors": 0, "warnings": 3, "infos": 1},
        }
        val_seg = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 0, "label2": 0},
                    "undefined_labels": {},
                }
            },
            "validation_reports": [
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeA"},
                {"anomaly_type": "TypeB"},
                {"anomaly_type": "TypeC"},
            ],
            "summary": {"errors": 0, "warnings": 3, "infos": 1},
        }

        result = get_tab_data_for_label_dist_by_type(val_cls, val_det, val_seg)

        expected_defined = [
            {
                "title": "All",
                "data": [
                    {"label": 5, "bbox": 0, "polygon": 0, "id": "label1"},
                    {"label": 3, "bbox": 0, "polygon": 0, "id": "label2"},
                ],
                "chart_type": Dashboard.Chart.Radar,
                "chart_kwargs": {"indexBy": "id", "keys": {"label", "polygon", "bbox"}},
            },
            {
                "title": "Label",
                "data": [
                    {"id": "label1", "label": "label1", "value": 5},
                    {"id": "label2", "label": "label2", "value": 3},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
            {
                "title": "Bbox",
                "data": [
                    {"id": "label1", "label": "label1", "value": 0},
                    {"id": "label2", "label": "label2", "value": 0},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
            {
                "title": "Polygon",
                "data": [
                    {"id": "label1", "label": "label1", "value": 0},
                    {"id": "label2", "label": "label2", "value": 0},
                ],
                "chart_type": Dashboard.Chart.Pie,
            },
        ]
        expected_undefined = [
            {
                "title": "All",
                "data": [{"label": 2, "id": "label3"}],
                "chart_type": Dashboard.Chart.Radar,
                "chart_kwargs": {"indexBy": "id", "keys": {"label"}},
            },
            {
                "title": "Label",
                "data": [{"id": "label3", "label": "label3", "value": 2}],
                "chart_type": Dashboard.Chart.Pie,
            },
        ]

        assert result == (expected_defined, expected_undefined)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_radar_data_for_label_dist_by_type(self):
        """Test if get_radar_data_for_label_dist_by_type function of analyze component is worked out correctly."""
        val_cls = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 2, "label2": 5},
                    "undefined_labels": {"label3": 3},
                }
            }
        }
        val_det = {
            "statistics": {
                "label_distribution": {
                    "defined_labels": {"label1": 1},
                    "undefined_labels": {"label4": 4},
                }
            }
        }
        val_seg = {}  # Empty

        result = get_radar_data_for_label_dist_by_type(val_cls, val_det, val_seg)

        expected_defined = [
            {"id": "label1", "label": 2, "bbox": 1},
            {"id": "label2", "label": 5},
        ]
        expected_keys_defined = {"label", "bbox"}
        expected_undefined = [
            {"id": "label3", "label": 3},
            {"id": "label4", "bbox": 4},
        ]
        expected_keys_undefined = {"label", "bbox"}

        assert result == (
            expected_defined,
            expected_keys_defined,
            expected_undefined,
            expected_keys_undefined,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_tab_data_point_dist_in_label(self):
        """Test if get_tab_data_point_dist_in_label function of analyze component is worked out correctly."""
        val_report = {
            "statistics": {
                "point_distribution_in_label": {
                    "label1": {
                        "area(wxh)": {"mean": 10, "median": 8},
                        "ratio(w/h)": {"mean": 1.5, "median": 1.2},
                    },
                    "label2": {
                        "area(wxh)": {"mean": 20, "median": 15},
                        "ratio(w/h)": {"mean": 2.0, "median": 1.8},
                    },
                }
            }
        }

        expected_tabs = [
            {
                "title": "area",
                "data": [
                    {"id": "label1", "mean": 10, "median": 8},
                    {"id": "label2", "mean": 20, "median": 15},
                ],
                "chart_type": Dashboard.Chart.Bar,
                "chart_kwargs": {
                    "indexBy": "id",
                    "keys": ["mean", "median"],
                    "groupMode": "grouped",
                    "valueFormat": ">-,.2f",
                },
            },
            {
                "title": "ratio",
                "data": [
                    {"id": "label1", "mean": 1.5, "median": 1.2},
                    {"id": "label2", "mean": 2.0, "median": 1.8},
                ],
                "chart_type": Dashboard.Chart.Bar,
                "chart_kwargs": {
                    "indexBy": "id",
                    "keys": ["mean", "median"],
                    "groupMode": "grouped",
                    "valueFormat": ">-,.2f",
                },
            },
        ]

        result = get_tab_data_point_dist_in_label(val_report)

        assert result == expected_tabs
