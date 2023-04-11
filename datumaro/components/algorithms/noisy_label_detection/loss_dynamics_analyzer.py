# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.dataset_base import IDataset
from datumaro.errors import DatasetError

__all__ = ["LossDynamicsAnalyzer", "NoisyLabelCandidate"]


@dataclass(order=True, frozen=True)
class NoisyLabelCandidate:
    id: str = field(compare=False)
    subset: str = field(compare=False)
    ann_id: int = field(compare=False)
    label_id: int = field(compare=False)
    metric: float = field(compare=True)

    def __eq__(self, __o: "NoisyLabelCandidate") -> bool:
        return self.id == __o.id and self.subset == __o.subset and self.ann_id == __o.ann_id


class LossDynamicsAnalyzer:
    """A class for analyzing the dynamics of training loss to identify noisy labels.

    This class parses the dataset to extract information about the training loss dynamics.
    It then calculates the exponential moving average (EMA) of the training loss dynamics.
    A higher EMA value of training loss dynamics can indicate a noisy labeled sample [1]_.
    The class provides an interface to extract the top-k candidates for noisy labels
    based on the statistics. Additionally, it can plot the EMA curves of loss dynamics for the candidates,
    allowing comparison of the dataset's overall average or averages grouped by labels.

    .. [1] Zhou, Tianyi, Shengjie Wang, and Jeff Bilmes.
    "Robust curriculum learning: from clean label detection to noisy label self-correction."
    International Conference on Learning Representations. 2021.
    """

    allowed_task_names = {"OTX-MultiClassCls"}

    def __init__(self, dataset: IDataset, alpha: float = 0.001) -> None:
        purpose = dataset.infos().get("purpose")
        if purpose != "noisy_label_detection":
            raise DatasetError(
                f'Dataset infos should have purpose="noisy_label_detection", but it has {purpose}.'
            )

        task = dataset.infos().get("task")
        if task != "OTX-MultiClassCls":
            raise DatasetError(
                f"{task} is not allowed. The allowed task names are {self.allowed_task_names}."
            )

        self._dataset = dataset
        self._alpha = alpha
        self._df = self._parse_to_dataframe(dataset, alpha)
        self._mean_loss_dyns = self._df.mean(0)
        self._mean_loss_dyns_per_label = {
            label_id: df_sub.mean(0) for label_id, df_sub in self._df.groupby("ann_label")
        }

    @property
    def alpha(self) -> float:
        """A parameter to obtain EMA loss dynamics statistics.

        ema_loss_dyns(t) := (1 - alpha) * ema_loss_dyns(t - 1) + alpha * loss_dyns(t)"""
        return self._alpha

    @property
    def mean_loss_dyns(self) -> pd.Series:
        """Pandas Series object obtained by averaging all EMA loss dynamics statistics"""
        return self._mean_loss_dyns

    @property
    def mean_loss_dyns_per_label(self) -> Dict[LabelCategories.Category, pd.Series]:
        """A dictionary of Pandas Series object obtained
        by averaging EMA loss dynamics statistics according to the label category"""
        label_categories = self._dataset.categories()[AnnotationType.label]
        return {label_categories[k]: v for k, v in self._mean_loss_dyns_per_label.items()}

    @property
    def ema_dataframe(self) -> pd.DataFrame:
        """Pandas DataFrame including full EMA loss dynamics statistics."""
        return self._df

    @staticmethod
    def _parse_to_dataframe(dataset: IDataset, ema_alpha: float = 0.001) -> pd.DataFrame:
        """Parse loss dynamics statistics from Datumaro dataset to Pandas DataFrame."""
        ema_loss_dyns_list = []
        for item in dataset:
            for ann in item.annotations:
                # The first value should be start from zero
                loss_dyns = pd.Series(
                    [0.0] + ann.attributes.get("loss_dynamics"),
                    index=[-1] + ann.attributes.get("iters"),
                    name=(item.id, item.subset, ann.id, ann.label),
                )
                ema_loss_dyns = loss_dyns.ewm(alpha=ema_alpha, adjust=False).mean().iloc[1:]
                ema_loss_dyns_list.append(ema_loss_dyns)

        df = pd.DataFrame(ema_loss_dyns_list).rename_axis(
            ["item_id", "item_subset", "ann_id", "ann_label"]
        )
        df.sort_index(axis=1, inplace=True)
        df["last_value"] = df.apply(lambda row: row[row.last_valid_index()], axis=1)
        df.insert(0, "last_value", df.pop("last_value"))
        df.sort_values(by="last_value", inplace=True)

        return df

    def get_top_k_cands(self, top_k: int) -> List[NoisyLabelCandidate]:
        """Return a list of top-k noisy label candidates.

        Parameters
        ----------
        top_k: int
            An integer value to determine the number of candidates

        Returns
        -------
            A list of top-k noisy label candidates.
            It is sorted in descending order by the last value of the EMA training loss dynamics.
        """
        return [
            NoisyLabelCandidate(item_id, item_subset, ann_id, label_id, value)
            for (item_id, item_subset, ann_id, label_id), value in self._df["last_value"]
            .iloc[-top_k:]
            .items()
        ][::-1]

    def plot_ema_loss_dynamics(
        self,
        candidates: Sequence[NoisyLabelCandidate],
        mode: str = "mean",
        mean_plot_style: str = "--",
        mean_plot_color: str = "k",
        figsize: Tuple[int, int] = (4, 3),
        **kwargs,
    ) -> Figure:
        if mode == "mean":
            cands_by_label_id = {None: candidates}
        elif mode == "label_mean":
            cands_by_label_id = defaultdict(list)
            for cand in candidates:
                cands_by_label_id[cand.label_id].append(cand)
        else:
            raise ValueError(f'mode={mode} is not allowed, it should be "mean" or "label_mean".')

        label_categories = self._dataset.categories()[AnnotationType.label]
        n_cols = len(cands_by_label_id)
        fig, _ = plt.subplots(1, n_cols, figsize=(n_cols * figsize[0], figsize[1]))

        for idx, label_id in enumerate(sorted(list(cands_by_label_id.keys()))):
            plt.subplot(1, n_cols, idx + 1)

            # We have to drop 0 column since it is "last_value" column.
            mean_series = (
                self._mean_loss_dyns.iloc[1:]
                if label_id is None
                else self._mean_loss_dyns_per_label[label_id].iloc[1:]
            )
            plt.plot(mean_series, color=mean_plot_color, linestyle=mean_plot_style, label="mean")

            cands = cands_by_label_id[label_id]
            ids = [cand.id for cand in cands]
            target = self._df.query(f"item_id == {ids}")

            for (item_id, subset_id, ann_id, label), row in target.iterrows():
                plt.plot(
                    row.dropna().iloc[1:],
                    label=f"id={item_id}, subset={subset_id}, ann_id={ann_id}, label_id={label}",
                    **kwargs,
                )

            plt.ylabel("EMA training loss dynamics")
            plt.xlabel("Training iterations")

            plt.title(
                f"Compare the candidates with the average of ({label_categories[label_id].name})"
                if label_id is not None
                else "Compare the candidates with the average of dataset"
            )
            plt.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1
            )

        return fig
