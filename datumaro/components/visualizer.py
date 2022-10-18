# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import warnings
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset import IDataset
from datumaro.components.extractor import DatasetItem

DEFAULT_COLOR_CYCLES: List[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class Visualizer:
    def __init__(
        self,
        dataset: IDataset,
        ignored_types: Optional[Iterable[AnnotationType]] = None,
        figsize: Tuple[float, float] = (8, 6),
        color_cycles: Optional[List[str]] = None,
        bbox_linewidth: float = 1.0,
        text_y_offset: float = 1.5,
    ) -> None:
        self.dataset = dataset
        self.figsize = figsize
        self.ignored_types = set(ignored_types) if ignored_types is not None else set()
        self.color_cycles = color_cycles if color_cycles is not None else DEFAULT_COLOR_CYCLES
        self.bbox_linewidth = bbox_linewidth
        self.text_y_offset = text_y_offset

    def _sort_by_z_order(self, annotations: List[Annotation]) -> List[Annotation]:
        def _sort_key(ann: Annotation):
            z_order = getattr(ann, "z_order", -1)
            return z_order

        return sorted(annotations, key=_sort_key)

    def vis_gallery(
        self,
        ids: List[Union[str, DatasetItem]],
        nrows: int,
        ncols: int,
        subset: Optional[str] = None,
    ) -> Figure:
        assert nrows > 0, "nrows should be a positive integer."
        assert ncols > 0, "ncols should be a positive integer."
        assert (
            len(ids) <= nrows * ncols
        ), "The number of ids should less then or equal to nrows * ncols."

        fig, axs = plt.subplots(nrows, ncols, figsize=self.figsize)

        for dataset_id, ax in zip(ids, axs.flatten()):
            self.vis_one_sample(dataset_id, subset, ax)

        return fig

    def vis_one_sample(
        self, id: Union[str, DatasetItem], subset: Optional[str] = None, ax: Optional[Axes] = None
    ) -> Figure:
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = plt.gca()
        else:
            fig = ax.get_figure()
            plt.sca(ax)

        item: DatasetItem = self.dataset.get(id, subset)
        assert item is not None, f"Cannot find id={id}, subset={subset}"

        img = item.media.data.astype(np.uint8)
        ax.imshow(img)

        annotations = self._sort_by_z_order(item.annotations)
        categories = self.dataset.categories()
        label_categories = (
            self.dataset.categories()[AnnotationType.label]
            if AnnotationType.label in categories
            else None
        )

        for ann in annotations:
            if ann.type in self.ignored_types:
                warnings.warn(f"{ann.type} in self.ignored_types. Skip it.")
                continue

            if ann.type == AnnotationType.bbox:
                self._draw_bbox(ann, label_categories, ax)
            else:
                raise NotImplementedError(f"{ann.type} is not implemented yet.")

        ax.set_title(f"ID: {id}, Subset={subset}")
        ax.set_axis_off()

        return fig

    def _draw_bbox(self, ann: Bbox, label_categories: Optional[LabelCategories], ax: Axes):
        label_text = label_categories[ann.label].name if label_categories is not None else ann.label
        color = self.color_cycles[ann.label % len(self.color_cycles)]
        rect = patches.Rectangle(
            (ann.x, ann.y),
            ann.w,
            ann.h,
            linewidth=self.bbox_linewidth,
            edgecolor=color,
            facecolor="none",
        )
        ax.text(ann.x, ann.y - self.text_y_offset, label_text, color=color)
        ax.add_patch(rect)
