# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import abc
from collections import defaultdict
from types import SimpleNamespace
from typing import NamedTuple, NewType

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.components.single.tabs.transform import TransformBase
from datumaro_gui.utils.dataset.data_loader import MultipleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, Pie, Radar
from datumaro_gui.utils.drawing.css import box_style
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro import AnnotationType
from datumaro.components.visualizer import Visualizer


class MultipleTransformBase(TransformBase):
    @abc.abstractclassmethod
    def gui(self, data_helper: MultipleDatasetHelper, col: str):
        raise NotImplementedError


class TransformLabelRemap(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Label Remapping"

    @property
    def info(self) -> str:
        return "This helps to remap labels of dataset."

    def _do_label_remap(
        self, data_helper, mapping, uploaded_file_2, uploaded_file_1, mode, delete_unselected, col
    ):
        mapping_dict = (
            dict(zip(mapping[uploaded_file_2], mapping[uploaded_file_1]))
            if mode == "reverse"
            else dict(zip(mapping[uploaded_file_1], mapping[uploaded_file_2]))
        )
        labels_set = {
            label.name for label in data_helper._dm_dataset.categories()[AnnotationType.label]
        }
        mapping_dict.update({label: "background" for label in labels_set - set(mapping_dict)})

        default = "delete" if delete_unselected else "keep"
        data_helper.transform("remap_labels", mapping=mapping_dict, default=default)
        st.toast("Remap Success!", icon="🎉")

    def gui(self, data_helper: MultipleDatasetHelper, col):
        uploaded_file_1 = state["uploaded_file_1"]
        uploaded_file_2 = state["uploaded_file_2"]
        mapping = (
            pd.DataFrame(columns=[uploaded_file_1, uploaded_file_2])
            if state.mapping is None or state.mapping.empty
            else state.mapping
        )

        st.warning("Please mapping category first") if mapping.empty else st.dataframe(
            mapping, use_container_width=True
        )
        mode = "default" if col == "c1" else "reverse"
        delete_unselected = st.toggle("Delete unselected labels", key=f"remap_del_tog_{col}")

        remap_btn = st.button("Do Label Remap", use_container_width=True, key=f"remap_btn_{col}")
        if remap_btn:
            self._do_label_remap(
                data_helper, mapping, uploaded_file_2, uploaded_file_1, mode, delete_unselected, col
            )


class TransformAggregation(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Aggregation"

    @property
    def info(self) -> str:
        return "This helps to merge subsets within a dataset into a single subset."

    def _do_aggregation(self, data_helper, selected_subsets, dst_subset_name):
        data_helper.aggregate(from_subsets=selected_subsets, to_subset=dst_subset_name)
        st.toast("Aggregation Success!", icon="🎉")

    def gui(self, data_helper: MultipleDatasetHelper, col):
        subsets = list(data_helper.dataset().subsets().keys())
        selected_subsets = st.multiselect(
            "Select subsets to be aggregated",
            subsets,
            default=subsets,
            key=f"aggre_subset_list_{col}",
        )
        dst_subset_name = st.text_input(
            "Aggregated Subset Name:", "default", key=f"aggre_subset_name_{col}"
        )
        aggre_btn = st.button(
            "Do aggregation",
            use_container_width=True,
            key=f"aggre_subset_btn_{col}",
        )
        if aggre_btn:
            self._do_aggregation(data_helper, selected_subsets, dst_subset_name)


class TransformSplit(MultipleTransformBase):
    class Split(NamedTuple):
        subset: str
        ratio: float

    @property
    def name(self) -> str:
        return "Split"

    @property
    def info(self) -> str:
        return "This helps to divide a dataset into multiple subsets with a given ratio."

    def _add_subset(self, col):
        subset_state = state[f"subset_{col[-1]}"]
        default_splits = (self.Split("train", 0.5), self.Split("val", 0.2), self.Split("test", 0.3))
        idx = 0  # default is 'train'
        default_names = tuple(split.subset for split in default_splits)
        for split in reversed(subset_state):
            if split.subset in default_names:
                idx = (default_names.index(split.subset) + 1) % len(default_names)
                break
        state[f"subset_{col[-1]}"].append(default_splits[idx])

    @staticmethod
    def _delete_subset(idx, col):
        state[f"subset_{col[-1]}"].pop(idx)

    def _do_split(self, data_helper, subset_state):
        ratios = [split.ratio for split in subset_state]
        total = sum(ratios)
        if total == 1:
            data_helper.transform("random_split", splits=subset_state)
            st.toast("Split Success!", icon="🎉")
        else:
            st.toast("Sum of ratios is expected to be 1!", icon="🚨")

    def gui(self, data_helper: MultipleDatasetHelper, col):
        c1, c2 = st.columns(2)
        add_subset_btn = c1.button(
            "Add subset",
            use_container_width=True,
            key=f"add_subset_btn_{col}",
        )
        if add_subset_btn:
            self._add_subset(col)

        subset_state = state[f"subset_{col[-1]}"]
        if len(subset_state) > 0:
            name, ratio, remove = st.columns([0.45, 0.45, 0.1])
            with name:
                st.write("Name")
            with ratio:
                st.write("Ratio")
            with remove:
                st.write("x")

            for idx, split in enumerate(subset_state):
                with name:
                    subset_name = st.text_input(
                        key=f"subset_name_{idx}_{col}",
                        label="name",
                        value=split.subset,
                        label_visibility="collapsed",
                    )
                with ratio:
                    subset_ratio = st.text_input(
                        key=f"subset_ratio_{idx}_{col}",
                        label="ratio",
                        value=split.ratio,
                        label_visibility="collapsed",
                    )
                with remove:
                    st.button(
                        key=f"subset_remove_{idx}_{col}",
                        label=":no_entry:",
                        on_click=self._delete_subset,
                        args=(idx, col),
                    )

                subset_state[idx] = self.Split(subset_name, float(subset_ratio))

            ratios = [split.ratio for split in subset_state]
            total = sum(ratios)
            if total != 1:
                st.toast("Sum of ratios is expected to be 1!", icon="🚨")

        with c2:
            split_btn = st.button(
                "Do split",
                use_container_width=True,
                key=f"split_btn_{col}",
            )
            if split_btn:
                self._do_split(data_helper, subset_state)


class TransformSubsetRename(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Subset Rename"

    @property
    def info(self) -> str:
        return "This helps to rename subset in dataset"

    @staticmethod
    def _remap_subset(data_helper, target_subset, target_name):
        mapping = {target_subset: target_name}
        data_helper.transform("map_subsets", mapping=mapping)
        st.toast("Rename Subset Success!", icon="🎉")

    def _display_subsets(self, data_helper):
        subsets = [data_helper.dataset().subsets().keys()]
        df = pd.DataFrame(subsets)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    def gui(self, data_helper: MultipleDatasetHelper, col):
        subsets = list(data_helper.dataset().subsets().keys())

        c1, c2 = st.columns(2)
        with c1:
            target_subset = st.selectbox(
                "Select a subset to rename", subsets, key=f"subset_rename_sb_{col}"
            )
        with c2:
            target_name = st.text_input("New subset name", key=f"subset_rename_input_{col}")

        subset_rename_btn = st.button(
            "Do Subset Rename", use_container_width=True, key=f"subset_rename_btn_{col}"
        )
        if subset_rename_btn:
            self._remap_subset(data_helper, target_subset, target_name)
            self._display_subsets(data_helper)


class TransformReindexing(MultipleTransformBase):
    class UniqueID(NamedTuple):
        subset: str
        id: str

    @property
    def name(self) -> str:
        return "Reindexing"

    @property
    def info(self) -> str:
        return "This helps to reidentify all items."

    def _reindex_with_index(self, data_helper, start_index):
        data_helper.transform("reindex", start=start_index)
        st.toast("Reindex Success!", icon="🎉")

    def _reindex_with_image(self, data_helper):
        data_helper.transform("id_from_image_name")
        st.toast("Reindex Success!", icon="🎉")

    def _display_unique_ids(self, data_helper):
        ids = [self.UniqueID(item.subset, item.id) for item in data_helper.dataset()]
        df = pd.DataFrame(ids)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    def gui(self, data_helper: MultipleDatasetHelper, col):
        col1, col2 = st.columns(2)
        start_index = col1.number_input(
            "start number",
            min_value=0,
            label_visibility="collapsed",
            key=f"item_reindex_input_{col}",
        )

        item_reindex_btn = col2.button(
            f"Set IDs from {start_index}",
            use_container_width=True,
            key=f"item_reindex_btn_{col}",
        )
        if item_reindex_btn:
            self._reindex_with_index(data_helper, start_index)

        item_media_name_btn = st.button(
            "Set IDs with media name",
            use_container_width=True,
            key=f"item_media_name_btn_{col}",
        )
        if item_media_name_btn:
            self._reindex_with_image(data_helper)

        if item_reindex_btn or item_media_name_btn:
            self._display_unique_ids(data_helper)


class TransformFiltration(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Filtration"

    @property
    def info(self) -> str:
        return "This helps to filter some items or annotations within a dataset."

    @property
    def link(self) -> str:
        return f"{self._datumaro_doc}/command-reference/context_free/filter.html"

    def _filter_dataset(self, data_helper, filter_expr, selected_mode):
        if not filter_expr:
            st.toast("Enter XML filter expression", icon="🚨")
            return None

        filter_args_dict = {
            "items": {},
            "annotations": {"filter_annotations": True},
            "items+annotations": {"filter_annotations": True, "remove_empty": True},
        }
        filter_args = filter_args_dict.get(selected_mode, None)

        try:
            data_helper.filter(filter_expr, filter_args)
            st.toast("Filter Success!", icon="🎉")
        except Exception as e:
            st.toast(f"Error: {repr(e)}", icon="🚨")

    def _display_xml_representation(self, data_helper, col):
        dataset = data_helper.dataset()
        if dataset is None or len(dataset) == 0:
            st.warning("No items are left in the dataset.")
            return

        keys = data_helper.subset_to_ids()
        if len(keys.keys()) > 1:
            selected_subset = st.selectbox(
                "Select a subset", options=sorted(keys.keys()), key=f"show_xml_subset_{col}"
            )
            selected_id = st.selectbox(
                "Select an item", options=keys[selected_subset], key=f"show_xml_item_{col}"
            )
        else:
            selected_subset = keys.keys()[0]
            selected_id = st.selectbox(
                "Select an item",
                options=keys[selected_subset],
                key=f"show_xml_mulsubset_item_{col}",
            )

        xml_str = data_helper.get_xml(selected_subset, selected_id)
        if xml_str:
            st.code(xml_str, language="xml")

    def gui(self, data_helper: MultipleDatasetHelper, col):
        mode, filter_ = st.columns([4, 6])
        with mode:
            selected_mode = st.selectbox(
                "Select filtering mode",
                ["items", "annotations", "items+annotations"],
                key=f"filter_selected_mode_{col}",
            )
        with filter_:
            filter_expr = st.text_input(
                "Enter XML filter expression ([XPATH](https://devhints.io/xpath))",
                disabled=False,
                placeholder='Eg. /item[subset="train"]',
                value=None,
                key=f"filter_expr_{col}",
            )
        if selected_mode == "items+annotations":
            st.warning("Dataset items with no annotations would be removed")

        show_xml = st.toggle("Show XML Representations", key=f"show_xml_{col}")
        if show_xml:
            self._display_xml_representation(data_helper, col)

        filter_btn = st.button(
            "Filter dataset",
            use_container_width=True,
            key=f"filter_btn_{col}",
        )
        if filter_btn:
            self._filter_dataset(data_helper, filter_expr, selected_mode)


class TransformRemove(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Remove"

    @property
    def info(self) -> str:
        return "This helps to remove some items or annotations within a dataset."

    def _remove_item(self, data_helper, selected_id, selected_subset):
        data_helper.transform(
            "remove_items",
            ids=[(selected_id, selected_subset)],
        )
        st.toast("Remove Success!", icon="🎉")

    def _remove_annotation(self, data_helper, selected_id, selected_subset, selected_ann_id):
        ids = [(selected_id, selected_subset)]
        if selected_ann_id != "All":
            ids.append((selected_ann_id,))

        data_helper.transform(
            "remove_annotations",
            ids=ids,
        )
        st.toast("Success!", icon="🎉")

    def gui(self, data_helper: MultipleDatasetHelper, col):
        keys = data_helper.subset_to_ids()
        c1, c2, c3 = st.columns(3)
        selected_subset = c1.selectbox(
            "Select a subset", options=sorted(keys.keys()), key=f"selected_subset_{col}"
        )
        selected_id = c2.selectbox(
            "Select an item", options=keys[selected_subset], key=f"selected_id_{col}"
        )
        dataset = data_helper.dataset()
        selected_item = dataset.get(selected_id, selected_subset)
        ann_ids = ["All"] + sorted({ann.id for ann in selected_item.annotations})
        selected_ann_id = c3.selectbox(
            "Select an annotation:", ann_ids, key=f"selected_ann_id_{col}"
        )

        bc1, bc2 = st.columns(2)
        rm_item_btn = bc1.button(
            "Remove item",
            use_container_width=True,
            key=f"rm_item_btn_{col}",
        )
        if rm_item_btn:
            self._remove_item(data_helper, selected_id, selected_subset)

        rm_ann_btn = bc2.button(
            "Remove annotation",
            use_container_width=True,
            key=f"rm_ann_btn_{col}",
        )
        if rm_ann_btn:
            self._remove_annotation(data_helper, selected_id, selected_subset, selected_ann_id)

        num_items = len(dataset)
        rc1, rc2 = st.columns([0.3, 0.7])
        with rc1:
            st.metric("Items in the dataset", num_items)
            st.metric("Annotation in the item", len(selected_item.annotations))

        with rc2:
            if selected_item is None:
                st.warning("Selected item is removed from the dataset")
            else:
                visualizer = Visualizer(dataset, figsize=(8, 8), alpha=0.5, show_plot_title=False)

                if selected_ann_id == "All":
                    fig = visualizer.vis_one_sample(selected_item)
                else:
                    fig = visualizer.vis_one_sample(selected_item, ann_id=selected_ann_id)

                fig.set_facecolor("none")
                st.pyplot(fig, use_container_width=True)


SummaryType = NewType("SummaryType", dict[dict[str, int]])


class TransformAutoCorrection(MultipleTransformBase):
    @property
    def name(self) -> str:
        return "Auto-correction"

    @property
    def info(self) -> str:
        return "This helps to correct a dataset and clean up validation report."

    @property
    def link(self) -> str:
        return f"{self._datumaro_doc}/jupyter_notebook_examples/notebooks/12_correct_dataset.html"

    @staticmethod
    @st.cache_data
    def _recommend_task(stats_anns: dict) -> str:
        anns_by_type = {
            type: stats_anns.get("annotations by type", {}).get(type, {}).get("count", 0)
            for type in ["label", "bbox", "polygon", "mask", "ellipse"]
        }

        num_cls, num_det, num_seg = (
            anns_by_type["label"],
            anns_by_type["bbox"],
            (anns_by_type["polygon"] + anns_by_type["mask"] + anns_by_type["ellipse"]),
        )

        num_max = max((num_cls, num_det, num_seg))
        return (
            "Classification"
            if num_max == num_cls
            else "Detection"
            if num_max == num_det
            else "Segmentation"
        )

    @staticmethod
    @st.cache_data
    def _get_validation_summary(reports) -> SummaryType | None:
        if len(reports["validation_reports"]) == 0:
            return None

        summary = defaultdict(dict)
        for report in reports["validation_reports"]:
            anomaly_type = report["anomaly_type"]
            severity = report["severity"]
            summary[severity][anomaly_type] = summary[severity].get(anomaly_type, 0) + 1
        return summary

    @staticmethod
    @st.cache_data
    def _get_df(summary: SummaryType) -> pd.DataFrame:
        df_items = []
        severities = ["error", "warning", "info"]
        for severity in severities:
            anomalies = summary.get(severity, {})
            df_items.extend(
                [
                    {"severity": severity, "anomaly_type": anomaly, "count": count}
                    for anomaly, count in anomalies.items()
                ]
            )
        return pd.DataFrame(df_items)

    @staticmethod
    @st.cache_data
    def _get_compared_df(summary1: SummaryType, summary2: SummaryType) -> pd.DataFrame:
        df_items = []
        severities = ["error", "warning", "info"]
        for severity in severities:
            anomalies1, anomalies2 = summary1.get(severity, {}), summary2.get(severity, {})
            anomalies = set(anomalies1.keys()) | set(anomalies2.keys())

            df_items.extend(
                [
                    {
                        "severity": severity,
                        "anomaly_type": anomaly,
                        "count(src)": anomalies1.get(anomaly, 0),
                        "count(dst)": anomalies2.get(anomaly, 0),
                    }
                    for anomaly in sorted(anomalies)
                ]
            )

        return pd.DataFrame(df_items)

    @staticmethod
    def _correct_dataset(data_helper, selected_task, col):
        try:
            reports_src = data_helper.validate(selected_task)
            result = data_helper.transform("correct", reports=reports_src)
            reports_dst = data_helper.validate(selected_task)
            correct_reports = {"src": reports_src, "dst": reports_dst}
            state[f"correct_reports_{col[-1]}"] = correct_reports
            st.toast("Correction Success!", icon="🎉")
            return result
        except Exception as e:
            st.toast(f"Error: {repr(e)}", icon="🚨")
            return None

    def gui(self, data_helper: MultipleDatasetHelper, col):
        tasks = ["Classification", "Detection", "Segmentation"]
        recommended_task = self._recommend_task(data_helper.get_ann_stats())
        selected_task = st.selectbox(
            "Select a task", tasks, index=tasks.index(recommended_task), key=f"correct_task_{col}"
        )

        correct_btn = st.button(
            "Correct a dataset",
            use_container_width=True,
            key=f"correct_btn_{col}",
        )
        if correct_btn:
            self._correct_dataset(data_helper, selected_task, col)

        correct_reports = state[f"correct_reports_{col[-1]}"]
        if correct_reports is not None:
            summary_src = self._get_validation_summary(correct_reports["src"])
            summary_dst = self._get_validation_summary(correct_reports["dst"])
            st.dataframe(
                self._get_compared_df(summary_src, summary_dst),
                use_container_width=True,
                hide_index=True,
            )
            state[f"correct_reports_{col[-1]}"] = None  # reset state
        else:
            reports = data_helper.validate(selected_task)
            summary = self._get_validation_summary(reports)
            st.dataframe(self._get_df(summary), use_container_width=True, hide_index=True)
        return data_helper._dm_dataset


def render_dataset_management_section(
    col_name,
    data_helper,
):
    with st.expander("Category Management"):
        ######## Remap
        sac.divider(
            label="Label remapping", icon="map", align="center", bold=False, key=f"remap_{col_name}"
        )
        TransformLabelRemap().gui(data_helper=data_helper, col=col_name)

    with st.expander("Subset Management"):
        ######## Aggregation
        sac.divider(
            label="Aggregation", icon="columns", align="center", bold=False, key=f"aggre_{col_name}"
        )
        TransformAggregation().gui(data_helper, col_name)

        ######## Split
        sac.divider(
            label="Split", icon="columns-gap", align="center", bold=False, key=f"split_{col_name}"
        )
        TransformSplit().gui(data_helper, col_name)

        ######## Subset Rename
        sac.divider(
            label="Subset Rename",
            icon="columns-gap",
            align="center",
            bold=False,
            key=f"subset_rename_{col_name}",
        )
        TransformSubsetRename().gui(data_helper, col_name)

    with st.expander("Item Management"):
        ######## Reindex
        sac.divider(
            label="Reindexing",
            icon="stickies-fill",
            align="center",
            bold=False,
            key=f"reindex_{col_name}",
        )
        TransformReindexing().gui(data_helper, col_name)

        ######## Filter
        sac.divider(
            label="Filtration",
            icon="funnel-fill",
            align="center",
            bold=False,
            key=f"filter_{col_name}",
        )
        TransformFiltration().gui(data_helper, col_name)

        ######## Remove
        sac.divider(
            label="Remove", icon="eraser-fill", align="center", bold=False, key=f"remove_{col_name}"
        )
        TransformRemove().gui(data_helper, col_name)

        ######## Auto correction
        sac.divider(
            label="Auto-correction",
            icon="hammer",
            align="center",
            bold=False,
            key=f"auto_correct_{col_name}",
        )
        TransformAutoCorrection().gui(data_helper, col_name)


def main():
    data_helper_1: MultipleDatasetHelper = state["data_helper_1"]
    data_helper_2: MultipleDatasetHelper = state["data_helper_2"]
    uploaded_file_1 = state["uploaded_file_1"]
    uploaded_file_2 = state["uploaded_file_2"]

    dataset_dict = {uploaded_file_1: data_helper_1, uploaded_file_2: data_helper_2}
    subset_key_dict = {uploaded_file_1: "subset_1", uploaded_file_2: "subset_2"}
    report_key_dict = {uploaded_file_1: "report_1", uploaded_file_2: "report_2"}

    dataset_names = [uploaded_file_1, uploaded_file_2]
    if "data_helper_merged" in state:
        data_helper_3: MultipleDatasetHelper = state["data_helper_merged"]
        dataset_names.append("Merged Dataset")
        dataset_dict["Merged Dataset"] = data_helper_3
        subset_key_dict["Merged Dataset"] = "subset_merged"
        report_key_dict["Merged Dataset"] = "report_merged"

    c1, c2 = st.columns(2)
    st.markdown("<style>{}</style>".format(box_style), unsafe_allow_html=True)
    with c1:
        selected_dataset_1 = st.selectbox(
            "Select dataset to transform : ", dataset_names, index=0, key="trans_selectbox_d1"
        )

        data_helper_1 = dataset_dict.get(selected_dataset_1, None)

        render_dataset_management_section(
            "c1",
            data_helper_1,
        )

        c1.markdown(
            f"<div class='highlight blue box'><span class='bold'>{selected_dataset_1}</span></div>",
            unsafe_allow_html=True,
        )

    with c2:
        selected_dataset_2 = st.selectbox(
            "Select dataset to transform : ", dataset_names, index=1, key="trans_selectbox_d2"
        )
        data_helper_2 = dataset_dict.get(selected_dataset_2, None)

        render_dataset_management_section(
            "c2",
            data_helper_2,
        )

        c2.markdown(
            f"<div class='highlight red box'><span class='bold'>{selected_dataset_2}</span></div>",
            unsafe_allow_html=True,
        )

    with elements("transform"):
        dataset_1 = data_helper_1.dataset()
        subset_info_dict_1 = get_subset_info(dataset_1)
        categories_1 = dataset_1.categories()[AnnotationType.label]
        cat_info_dict_1 = get_category_info(dataset_1, categories_1)

        dataset_2 = data_helper_2.dataset()
        subset_info_dict_2 = get_subset_info(dataset_2)
        categories_2 = dataset_2.categories()[AnnotationType.label]
        cat_info_dict_2 = get_category_info(dataset_2, categories_2)

        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            subset_info_1=Pie(
                name="Subset info of Dataset 1",
                **{"board": board, "x": 0, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            cat_info_1=Radar(
                name="Category info of Dataset 1",
                indexBy="subset",
                keys=[cat.name for cat in categories_1.items],
                **{"board": board, "x": 3, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            subset_info_2=Pie(
                name="Subset info of Dataset 2",
                **{"board": board, "x": 6, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
            cat_info_2=Radar(
                name="Category info of Dataset 2",
                indexBy="subset",
                keys=[cat.name for cat in categories_2.items],
                **{"board": board, "x": 9, "y": 0, "w": 3, "h": 6, "minW": 2, "minH": 4},
            ),
        )

        with w.dashboard(rowHeight=50):
            w.subset_info_1(subset_info_dict_1)
            w.subset_info_2(subset_info_dict_2)
            w.cat_info_1(cat_info_dict_1)
            w.cat_info_2(cat_info_dict_2)
