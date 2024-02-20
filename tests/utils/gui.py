# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


def compare_init_state(orig_keys, st_keys):
    return all(item in st_keys for item in orig_keys)


def compare_multiple_datahelper(data_helper, expected):
    assert data_helper._dataset_dir == expected[0]
    assert data_helper._detected_formats == expected[1]
    assert data_helper._format == expected[2]

    # Check _init_dependent_variables
    assert data_helper._image_stats == None
    assert data_helper._image_size_info == None
    assert data_helper._ann_stats == None
    assert data_helper._val_reports == {}


def compare_single_datahelper(data_helper, expected):
    assert data_helper._dataset_dir == expected[0]
    assert data_helper._detected_formats == expected[1]
    assert data_helper._format == expected[2]


def compare_single_stats(data_helper, expected):
    # Check _init_dependent_variables
    assert data_helper._image_stats["dataset"] == expected
    assert data_helper._image_stats["subsets"]["train"]["images count"] == 2
    assert data_helper._image_stats["subsets"]["test"]["images count"] == 1
    assert data_helper._image_stats["subsets"]["validation"]["images count"] == 1

    assert data_helper._ann_stats["images count"] == 4
    assert data_helper._ann_stats["annotations count"] == 7
    assert data_helper._ann_stats["unannotated images count"] == 1
    assert data_helper._ann_stats["unannotated images"] == ["d"]
    assert (
        "label" in data_helper._ann_stats["annotations by type"]
        and data_helper._ann_stats["annotations by type"]["label"].get("count") == 7
    )

    assert data_helper._image_size_info
    assert data_helper._val_reports == {}
