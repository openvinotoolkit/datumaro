# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType, LabelCategories

from ..dashboard import Dashboard, DatasetInfoBox, Gallery, Pie, Radar
from ..data_loader import DatasetHelper
from .analyze import get_dataset_info


def main():
    data_helper: DatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    n_labels = len(dataset.categories().get(AnnotationType.label, LabelCategories()))
    stats_image = data_helper.get_image_stats()
    stats_anns = data_helper.get_ann_stats()
    image_size_info = data_helper.get_image_size_info()
    image_mean = image_size_info["image_size"]["mean"]

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
                try:
                    label_name = categories[ann.label].name
                    cat_info[item.subset][label_name] += 1
                except Exception:
                    pass

        cat_info_dict = []
        for subset, cats in cat_info.items():
            cats.update({"subset": subset})
            cat_info_dict.append(cats)

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            dataset_info=DatasetInfoBox(
                **{"board": board, "x": 0, "y": 0, "w": 4, "h": 8, "minW": 3, "minH": 3}
            ),
            subset_info=Pie(
                name="Subset info",
                **{"board": board, "x": 4, "y": 0, "w": 4, "h": 8, "minW": 3, "minH": 3},
            ),
            cat_info=Radar(
                name="Category info",
                indexBy="subset",
                keys=[cat.name for cat in categories.items],
                **{"board": board, "x": 8, "y": 0, "w": 4, "h": 8, "minW": 3, "minH": 3},
            ),
            player=Gallery(board, 0, 8, 12, 12, minH=3),
        )
        print(board)

        with w.dashboard(rowHeight=50):
            w.dataset_info(
                "Dataset Information",
                get_dataset_info(stats_image, stats_anns, image_mean, n_labels),
            )
            w.subset_info(subset_info_dict)
            w.cat_info(cat_info_dict)
            w.player(dataset)
