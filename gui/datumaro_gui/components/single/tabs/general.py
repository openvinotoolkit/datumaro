# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, DatasetInfoBox, Gallery, Pie, Radar
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro.components.annotation import AnnotationType

from .analyze import get_dataset_info


def main():
    data_helper: SingleDatasetHelper = state["data_helper"]
    dataset = data_helper.dataset()
    n_labels = data_helper.num_labels

    stats_image = data_helper.get_image_stats()  # state["stats_image"]
    stats_anns = data_helper.get_ann_stats()  # state["stats_anns"]
    image_size_info = data_helper.get_image_size_info()  # state["image_size_info"]

    image_mean = image_size_info["image_size"]["mean"]

    with elements("general"):
        subset_info_dict = get_subset_info(dataset)
        categories = dataset.categories()[AnnotationType.label]
        cat_info_dict = get_category_info(dataset, categories)

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

        with w.dashboard(rowHeight=50):
            w.dataset_info(
                "Dataset Information",
                get_dataset_info(stats_image, stats_anns, image_mean, n_labels),
            )
            w.subset_info(subset_info_dict)
            w.cat_info(cat_info_dict)
            w.player(dataset)
