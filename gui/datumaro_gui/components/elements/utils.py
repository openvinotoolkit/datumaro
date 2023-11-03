# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


def get_subset_info(dataset):
    subset_info_dict = []
    for subset in dataset.subsets():
        temp_dict = {
            "id": subset,
            "label": subset,
            "value": len(dataset.get_subset(subset)),
        }
        subset_info_dict.append(temp_dict)
    return subset_info_dict


def get_category_info(dataset, categories):
    cat_info = {s: {cat.name: 0 for cat in categories} for s in dataset.subsets()}
    for item in dataset:
        for ann in item.annotations:
            label_name = categories[ann.label].name
            cat_info[item.subset][label_name] += 1
    cat_info_dict = []
    for subset, cats in cat_info.items():
        cats.update({"subset": subset})
        cat_info_dict.append(cats)
    return cat_info_dict


# Define CSS styles for the boxes
box_style = """
    .highlight {
    border-radius: 0.4rem;
    color: white;
    padding: 0.5rem;
    margin-bottom: 1rem;
    }
    .bold {
    padding-left: 1rem;
    font-weight: 700;
    }
    .red {
    background-color: lightcoral;
    }
    .blue {
    background-color: lightblue;
    }
    .lightgrayish {
    background-color: #B0C4DE;
    }
    .lightgray {
    background-color: #E5E5E5;
    }
    .lightmintgreen {
    background-color: #A9DFBF;
    }
    .lightpurple {
    background-color: #C9A0DC;
    }
    .box {
    width: auto;
    max-width: 1000px;
    margin: "5px 5px 5px 5px";
    }
"""
