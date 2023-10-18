# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType

from ..dashboard import Dashboard, Gallery, Pie, Radar
from ..data_loader import DatasetHelper


def main():
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()

    with elements("general"):
        subset_info_dict = []
        for subset in dataset.subsets():
            temp_dict = {
                "id": subset,
                "label": subset,
                "value": len(dataset.get_subset(subset)),
            }
            subset_info_dict.append(temp_dict)

        categories = dataset.categories()[AnnotationType.label]
        subsets = dataset.subsets()
        cat_info = {s: {cat.name: 0 for cat in categories.items} for s in subsets}
        for item in dataset:
            for ann in item.annotations:
                label_name = categories[ann.label].name
                cat_info[item.subset][label_name] += 1

        cat_info_dict = []
        for subset, cats in cat_info.items():
            cats.update({"subset": subset})
            cat_info_dict.append(cats)

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            subset_info=Pie(
                name="Subset info",
                **{"board": board, "x": 0, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            cat_info=Radar(
                name="Category info",
                indexBy="subset",
                keys=[cat.name for cat in categories.items],
                **{"board": board, "x": 3, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            player=Gallery(board, 6, 0, 6, 12, minH=4),
        )

        with w.dashboard(rowHeight=50):
            w.subset_info(subset_info_dict)
            w.cat_info(cat_info_dict)
            w.player(dataset)
