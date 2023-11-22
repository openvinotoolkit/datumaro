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
            try:
                label_name = categories[ann.label].name
                cat_info[item.subset][label_name] += 1
            except Exception:
                pass
    cat_info_dict = []
    for subset, cats in cat_info.items():
        cats.update({"subset": subset})
        cat_info_dict.append(cats)
    return cat_info_dict
