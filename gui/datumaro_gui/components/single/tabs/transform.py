# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import abc
import logging as log
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, NamedTuple, NewType, Tuple, Union

import pandas as pd
import streamlit as st
import streamlit_antd_components as sac
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.dataset.info import get_category_info, get_subset_info
from datumaro_gui.utils.drawing import Dashboard, Gallery, Pie, Radar
from datumaro_gui.utils.page import check_image_backend
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit import session_state as state
from streamlit_elements import elements

from datumaro import AnnotationType, LabelCategories
from datumaro.components.visualizer import Visualizer


class TransformBase(metaclass=abc.ABCMeta):
    _datumaro_doc = "https://openvinotoolkit.github.io/datumaro/latest/docs"

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractclassmethod
    def info(self) -> str:
        raise NotImplementedError

    @property
    def link(self) -> str:
        return ""

    @abc.abstractclassmethod
    def gui(self, data_helper: SingleDatasetHelper):
        raise NotImplementedError


class TransformLabelRemap(TransformBase):
    @property
    def name(self) -> str:
        return "Label Remapping"

    @property
    def info(self) -> str:
        return "This helps to remap labels of dataset."

    @staticmethod
    def _do_label_remap(data_helper, selected_rows, delete_unselected):
        mapping_dict = {item["src"]: item["dst"] for item in selected_rows}
        default = "delete" if delete_unselected else "keep"
        data_helper.transform("remap_labels", mapping=mapping_dict, default=default)
        st.toast("Remap Success!", icon="🎉")

    def gui(self, data_helper: SingleDatasetHelper):
        dataset = data_helper.dataset()
        stats_anns = data_helper.get_ann_stats()
        labels: LabelCategories = dataset.categories().get(AnnotationType.label, LabelCategories())

        c1, c2 = st.columns([0.3, 0.7])
        with c1:
            cat_info_dict = get_category_info(dataset, labels)
            with elements("single-transform-label-remapping"):
                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    cat_info=Radar(
                        name="Category info",
                        indexBy="subset",
                        keys=[cat.name for cat in labels.items],
                        **{"board": board, "x": 0, "y": 0, "w": 4, "h": 4, "minW": 3, "minH": 3},
                    ),
                )
                with st.container():
                    with w.dashboard(rowHeight=100):
                        w.cat_info(cat_info_dict)
        with c2:
            hide_empty_labels = st.toggle(
                "Hide empty labels", value=True, key="tg_hide_empty_labels_sin"
            )
            if hide_empty_labels:
                label_names = []
                counts = []
                for key, val in stats_anns["annotations"]["labels"]["distribution"].items():
                    if val[0] > 0:
                        label_names.append(key)
                        counts.append(val[0])
            else:
                label_names = [l.name for l in labels]
                counts = []
                for label in label_names:
                    counts.append(stats_anns["annotations"]["labels"]["distribution"][label][0])
            mapping = pd.DataFrame({"src": label_names, "dst": label_names, "count": counts})
            state["mapping"] = mapping
            gb = GridOptionsBuilder.from_dataframe(mapping)
            gb.configure_pagination(enabled=True)
            gb.configure_selection(selection_mode="multiple", use_checkbox=True)
            gb.configure_column("dst", editable=True)
            gb.configure_grid_options(domLayout="normal")
            gridoptions = gb.build()
            grid_table = AgGrid(
                mapping,
                gridOptions=gridoptions,
                height=300,
                width="100%",
                update_mode=GridUpdateMode.MODEL_CHANGED,
                fit_columns_on_grid_load=True,
                theme="streamlit",
                custom_css={"#gridToolBar": {"padding-bottom": "0px !important"}},
            )
            delete_unselected = st.toggle(
                "Delete unselected labels", key="tg_delete_unselected_sin"
            )

            st.button(
                "Do Label Remap",
                use_container_width=True,
                on_click=self._do_label_remap,
                args=(data_helper, grid_table["selected_rows"], delete_unselected),
                key="btn_do_label_remap_sin",
            )


class TransformAggregation(TransformBase):
    @property
    def name(self) -> str:
        return "Aggregation"

    @property
    def info(self) -> str:
        return "This helps to merge subsets within a dataset into a single subset."

    @staticmethod
    def _do_aggregation(data_helper, selected_subsets, dst_subset_name):
        data_helper.aggregate(from_subsets=selected_subsets, to_subset=dst_subset_name)
        st.toast("Aggregation Success!", icon="🎉")

    def gui(self, data_helper: SingleDatasetHelper):
        log.info(f"{__class__} called")

        subsets = list(data_helper.dataset().subsets().keys())

        c1, c2 = st.columns([0.3, 0.7])
        with c1:
            subset_info = get_subset_info(data_helper.dataset())
            with elements("single-transform-aggregation"):
                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    subset_info=Pie(
                        name="Subset info",
                        **{"board": board, "x": 0, "y": 0, "w": 4, "h": 4, "minW": 3, "minH": 3},
                    ),
                )
                with st.container():
                    with w.dashboard(rowHeight=100):
                        w.subset_info(subset_info)
        with c2:
            selected_subsets = st.multiselect(
                "Select subsets to be aggregated",
                subsets,
                default=subsets,
                key="ms_selected_subsets_agg_sin",
            )
            dst_subset_name = st.text_input(
                "Aggregated Subset Name:", "default", key="ti_dst_subset_name_agg_sin"
            )
            btn_do_aggr = st.button(
                "Do aggregation",
                use_container_width=True,
                key="btn_agg_sin",
            )
            if btn_do_aggr:
                self._do_aggregation(data_helper, selected_subsets, dst_subset_name)


class TransformSplit(TransformBase):
    class Split(NamedTuple):
        subset: str
        ratio: float

    @property
    def name(self) -> str:
        return "Split"

    @property
    def info(self) -> str:
        return "This helps to divide a dataset into multiple subsets with a given ratio."

    def _add_subset(self):
        subset_state = state["subset"]
        default_splits = (self.Split("train", 0.5), self.Split("val", 0.2), self.Split("test", 0.3))
        idx = 0  # default is 'train'
        default_names = tuple(split.subset for split in default_splits)
        for split in reversed(subset_state):
            if split.subset in default_names:
                idx = (default_names.index(split.subset) + 1) % len(default_names)
                break
        state["subset"].append(default_splits[idx])

    @staticmethod
    def _delete_subset(idx):
        state["subset"].pop(idx)

    @staticmethod
    def _do_split(data_helper, subset_state):
        ratios = [split.ratio for split in subset_state]
        total = sum(ratios)
        if total == 1:
            data_helper.transform("random_split", splits=subset_state)
            st.toast("Split Success!", icon="🎉")
        else:
            st.toast("Sum of ratios is expected to be 1!", icon="🚨")

    def gui(self, data_helper: SingleDatasetHelper):
        log.info(f"{__class__} called")

        c1, c2 = st.columns(2)
        subset_state = state["subset"]

        btn_add_subset = c1.button(
            "Add subset", use_container_width=True, key="btn_add_subset_split_sin"
        )
        if btn_add_subset:
            self._add_subset()

        btn_split = c2.button("Do split", use_container_width=True, key="btn_do_split_sin")
        if btn_split:
            self._do_split(data_helper, subset_state)

        chart, name, ratio, remove = st.columns([0.3, 0.3, 0.3, 0.1])
        dataset = data_helper.dataset()
        subset_info = get_subset_info(dataset)
        with chart:
            with elements("single-transform-split"):
                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    subset_info=Pie(
                        name="Subset info",
                        **{"board": board, "x": 0, "y": 0, "w": 4, "h": 4, "minW": 3, "minH": 3},
                    ),
                )
                with st.container():
                    with w.dashboard():
                        w.subset_info(subset_info)

        if len(subset_state) > 0:
            with name:
                st.write("Name")
            with ratio:
                st.write("Ratio")
            with remove:
                st.write("x")

            for idx, split in enumerate(subset_state):
                with name:
                    subset_name = st.text_input(
                        key=f"ti_subset_name_split_sin_{idx}",
                        label="name",
                        value=split.subset,
                        label_visibility="collapsed",
                    )
                with ratio:
                    subset_ratio = st.text_input(
                        key=f"ti_subset_ratio_split_sin_{idx}",
                        label="ratio",
                        value=split.ratio,
                        label_visibility="collapsed",
                    )
                with remove:
                    st.button(
                        key=f"btn_subset_remove_split_sin_{idx}",
                        label=":no_entry:",
                        on_click=self._delete_subset,
                        args=(idx,),
                    )
                state["subset"][idx] = self.Split(subset_name, float(subset_ratio))

            ratios = [split.ratio for split in subset_state]
            total = sum(ratios)
            if total != 1:
                st.toast("Sum of ratios is expected to be 1!", icon="🚨")


class TransformSubsetRename(TransformBase):
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

    def gui(self, data_helper: SingleDatasetHelper):
        log.info(f"{__class__} called")

        subsets = list(data_helper.dataset().subsets().keys())
        c1, c2, c3 = st.columns([0.3, 0.4, 0.3])
        with c1:
            subset_info = get_subset_info(data_helper.dataset())
            with elements("single-transform-subset-rename"):
                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    subset_info=Pie(
                        name="Subset info",
                        **{"board": board, "x": 0, "y": 0, "w": 4, "h": 4, "minW": 3, "minH": 3},
                    ),
                )
                with st.container():
                    with w.dashboard(rowHeight=100):
                        w.subset_info(subset_info)
        with c2:
            target_subset = st.selectbox(
                "Select a subset to rename", subsets, key="sb_subset_rename_sin"
            )
        with c3:
            target_name = st.text_input("New subset name", key="ti_subset_rename_sin")

        btn_subset_rename = c2.button(
            "Do Subset Rename",
            use_container_width=True,
            key="btn_subset_rename_sin",
        )
        if btn_subset_rename:
            self._remap_subset(data_helper, target_subset, target_name)


class TransformReindexing(TransformBase):
    class UniqueID(NamedTuple):
        subset: str
        id: str

    @property
    def name(self) -> str:
        return "Reindexing"

    @property
    def info(self) -> str:
        return "This helps to reidentify all items."

    @staticmethod
    def _reindex_with_index(data_helper, start_index):
        data_helper.transform("reindex", start=start_index)
        st.toast("Reindex Success!", icon="🎉")

    @staticmethod
    def _reindex_with_image(data_helper):
        data_helper.transform("id_from_image_name")
        st.toast("Reindex Success!", icon="🎉")

    def gui(self, data_helper: SingleDatasetHelper):
        col1, col2 = st.columns(2)
        start_index = col1.number_input(
            "start number",
            min_value=0,
            label_visibility="collapsed",
            key="ni_start_idx_reindex_sin",
        )
        btn_reindex_idx = col2.button(
            f"Set IDs from {start_index}",
            use_container_width=True,
            key="btn_start_idx_reindex_sin",
        )
        if btn_reindex_idx:
            self._reindex_with_index(data_helper, start_index)

        btn_reindex_media_name = st.button(
            "Set IDs with media name",
            use_container_width=True,
            key="btn_start_media_name_reindex_sin",
        )
        if btn_reindex_media_name:
            self._reindex_with_image(
                data_helper,
            )

        ids = []
        for item in data_helper.dataset():
            ids.append(self.UniqueID(item.subset, item.id))

        df = pd.DataFrame(ids)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)


class TransformFiltration(TransformBase):
    @property
    def name(self) -> str:
        return "Filtration"

    @property
    def info(self) -> str:
        return "This helps to filter some items or annotations within a dataset."

    @property
    def link(self) -> str:
        return f"{self._datumaro_doc}/command-reference/context_free/filter.html"

    @staticmethod
    def _filter_dataset(data_helper, filter_expr, selected_mode):
        if filter_expr:
            filter_args_dict = {
                "items": {},
                "annotations": {"filter_annotations": True},
                "items+annotations": {
                    "filter_annotations": True,
                    "remove_empty": True,
                },
            }
            filter_args = filter_args_dict.get(selected_mode, None)
            try:
                data_helper.filter(
                    filter_expr, filter_args
                )  # dataset.filter(filter_expr, **filter_args)
                st.toast("Filter Success!", icon="🎉")
            except Exception as e:
                st.toast(f"Error: {repr(e)}", icon="🚨")

        else:
            st.toast("Enter XML filter expression", icon="🚨")

    def gui(self, data_helper: SingleDatasetHelper):
        mode, filter_ = st.columns([4, 6])
        with mode:
            selected_mode = st.selectbox(
                "Select filtering mode",
                ["items", "annotations", "items+annotations"],
                key="sb_selected_mode_filt_sin",
            )
        with filter_:
            filter_expr = st.text_input(
                "Enter XML filter expression ([XPATH](https://devhints.io/xpath))",
                disabled=False,
                placeholder='Eg. /item[subset="train"]',
                value=None,
                key="ti_filter_expr_filt_sin",
            )
        if selected_mode == "items+annotations":
            st.warning("Dataset items with no annotations would be removed")
        btn_filter = st.button(
            "Filter dataset", use_container_width=True, key="btn_filter_filt_sin"
        )
        if btn_filter:
            self._filter_dataset(data_helper, filter_expr, selected_mode)

        show_xml = st.toggle("Show XML Representations", key="tg_show_xml_filt_sin")
        if show_xml:
            dataset = data_helper.dataset()
            if dataset is None or len(dataset) == 0:
                st.warning("No items are left in the dataset.")
            else:
                keys = data_helper.subset_to_ids()
                if len(keys.keys()) > 1:
                    c1, c2 = st.columns(2)
                    selected_subset = c1.selectbox(
                        "Select a subset",
                        options=sorted(keys.keys()),
                        key="sb_selected_subset_filt_sin",
                    )
                    selected_id = c2.selectbox(
                        "Select an item",
                        options=keys[selected_subset],
                        key="sb_selected_id_filt_sin",
                    )
                else:
                    selected_subset = list(keys.keys())[0]
                    selected_id = st.selectbox(
                        "Select an item",
                        options=keys[selected_subset],
                        key="sb_selected_id_2_filt_sin",
                    )

                xml_str = data_helper.get_xml(selected_subset, selected_id)
                if xml_str:
                    st.code(xml_str, language="xml")


class TransformRemove(TransformBase):
    @property
    def name(self) -> str:
        return "Remove"

    @property
    def info(self) -> str:
        return "This helps to remove some items or annotations within a dataset."

    @staticmethod
    def _remove_item(data_helper, selected_id, selected_subset):
        data_helper.transform(
            "remove_items",
            ids=[
                (selected_id, selected_subset),
            ],
        )
        st.toast("Remove Success!", icon="🎉")

    @staticmethod
    def _remove_annotation(data_helper, selected_id, selected_subset, selected_ann_id):
        if selected_ann_id == "All":
            data_helper.transform(
                "remove_annotations",
                ids=[
                    (selected_id, selected_subset),
                ],
            )
        else:
            data_helper.transform(
                "remove_annotations",
                ids=[
                    (selected_id, selected_subset, selected_ann_id),
                ],
            )
        st.toast("Success!", icon="🎉")

    def gui(self, data_helper: SingleDatasetHelper):
        keys = data_helper.subset_to_ids()
        c1, c2, c3 = st.columns(3)
        selected_subset = c1.selectbox(
            "Select a subset", options=sorted(keys.keys()), key="sb_select_subset_remove_sin"
        )
        selected_id = c2.selectbox(
            "Select an item", options=keys[selected_subset], key="sb_select_id_remove_sin"
        )
        dataset = data_helper.dataset()
        selected_item = dataset.get(selected_id, selected_subset)
        ann_ids = [
            "All",
        ] + sorted(list({ann.id for ann in selected_item.annotations}))
        selected_ann_id = c3.selectbox("Select an annotation", ann_ids, key="sb_select_ann_id_sin")

        bc1, bc2 = st.columns(2)
        btn_remove_item = bc1.button(
            "Remove item", use_container_width=True, key="btn_remove_item_remove_sin"
        )
        if btn_remove_item:
            self._remove_item(data_helper, selected_id, selected_subset)
        btn_remove_annot = bc2.button(
            "Remove annotation", use_container_width=True, key="btn_remove_annot_remove_sin"
        )
        if btn_remove_annot:
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


SummaryType = NewType("SummaryType", Dict[str, int])


class TransformAutoCorrection(TransformBase):
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
        anns_by_type = {}
        for type in ["label", "bbox", "polygon", "mask", "ellipse"]:
            anns_by_type[type] = (
                stats_anns.get("annotations by type", {}).get(type, {}).get("count", 0)
            )
        log.info(f"Annotation by types: {anns_by_type}")
        num_cls = anns_by_type["label"]
        num_det = anns_by_type["bbox"]
        num_seg = anns_by_type["polygon"] + anns_by_type["mask"] + anns_by_type["ellipse"]

        log.info(f"Annotations for tasks - cls: {num_cls}, det: {num_det}, seg: {num_seg}")
        num_max = max((num_cls, num_det, num_seg))

        if num_max == num_cls:
            return "Classification"
        elif num_max == num_det:
            return "Detection"
        else:
            return "Segmentation"

    @staticmethod
    @st.cache_data
    def _get_validation_summary(reports) -> Union[SummaryType, None]:
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
            for anomaly, count in sorted(anomalies.items()):
                df_items.append({"severity": severity, "anomaly_type": anomaly, "count": count})

        return pd.DataFrame(df_items)

    @staticmethod
    @st.cache_data
    def _get_compared_df(summary1: SummaryType, summary2: SummaryType) -> pd.DataFrame:
        df_items = []
        severities = ["error", "warning", "info"]
        for severity in severities:
            anomalies1 = summary1.get(severity, {})
            anomalies2 = summary2.get(severity, {})
            anomalies = set(list(anomalies1.keys()) + list(anomalies2.keys()))
            for anomaly in sorted(anomalies):
                count1 = anomalies1.get(anomaly, 0)
                count2 = anomalies2.get(anomaly, 0)
                df_items.append(
                    {
                        "severity": severity,
                        "anomaly_type": anomaly,
                        "count(src)": count1,
                        "count(dst)": count2,
                    }
                )

        return pd.DataFrame(df_items)

    @staticmethod
    def _correct_dataset(data_helper, selected_task):
        try:
            reports_src = data_helper.validate(selected_task)
            data_helper.transform("correct", reports=reports_src)
            reports_dst = data_helper.validate(selected_task)
            state["correct_reports"] = {"src": reports_src, "dst": reports_dst}
            st.toast("Correction Success!", icon="🎉")
        except Exception as e:
            st.toast(f"Error: {repr(e)}", icon="🚨")

    def gui(self, data_helper: SingleDatasetHelper):
        tasks = ["Classification", "Detection", "Segmentation"]
        recommended_task = self._recommend_task(data_helper.get_ann_stats())
        selected_task = st.selectbox(
            "Select a task",
            tasks,
            index=tasks.index(recommended_task),
            key="sb_select_task_auto_corr_sin",
        )

        btn_correct = st.button(
            "Correct a dataset", use_container_width=True, key="btn_auto_corr_sin"
        )
        if btn_correct:
            self._correct_dataset(data_helper, selected_task)

        if state["correct_reports"] is not None:
            log.info(state["correct_reports"])
            summary_src = self._get_validation_summary(state["correct_reports"]["src"])
            summary_dst = self._get_validation_summary(state["correct_reports"]["dst"])
            st.dataframe(
                self._get_compared_df(summary_src, summary_dst),
                use_container_width=True,
                hide_index=True,
            )
            state["correct_reports"] = None  # reset state
        else:
            reports = data_helper.validate(selected_task)
            summary = self._get_validation_summary(reports)
            st.dataframe(self._get_df(summary), use_container_width=True, hide_index=True)


class TransformNDR(TransformBase):
    @property
    def name(self) -> str:
        return "Near Duplicate Removal"

    @property
    def info(self) -> str:
        return "This helps to remove near-duplicated images in a subset"

    @staticmethod
    def _run_ndr(
        data_helper, working_subset, duplicated_subset, num_cut, over_sample, under_sample
    ):
        try:
            result = data_helper.transform(
                "ndr",
                working_subset=working_subset,
                duplicated_subset=duplicated_subset,
                num_cut=num_cut,
                over_sample=over_sample,
                under_sample=under_sample,
            )
            try:
                duplicated = result.get_subset(duplicated_subset)
            except KeyError:
                duplicated = []
            if len(duplicated) > 0:
                st.toast("NDR Success!", icon="🎉")
            else:
                st.toast("No duplication found!", icon="⚠️")
        except Exception as e:
            st.toast(f"Error: {repr(e)}", icon="🚨")

    @staticmethod
    def _remove_duplicated(data_helper, duplicated_subset):
        try:
            duplicated = data_helper.dataset().get_subset(duplicated_subset)
        except KeyError:
            st.toast("No items to remove!", icon="⚠️")
            return

        ids = []
        for item in duplicated:
            ids.append((item.id, duplicated_subset))

        try:
            data_helper.transform("remove_items", ids=ids)
            st.toast("Removal Success!", icon="🎉")
        except Exception as e:
            st.toast(f"Error: {repr(e)}", icon="🚨")

    def gui(self, data_helper: SingleDatasetHelper):
        subsets = list(data_helper.dataset().subsets().keys())

        c1, c2 = st.columns([0.3, 0.7])
        with c1:
            subset_info = get_subset_info(data_helper.dataset())
            with elements("single-transform-ndr-subset"):
                board = Dashboard()
                w = SimpleNamespace(
                    dashboard=board,
                    subset_info=Pie(
                        name="Subset info",
                        **{"board": board, "x": 0, "y": 0, "w": 4, "h": 4, "minW": 3, "minH": 3},
                    ),
                )
                with st.container():
                    with w.dashboard(rowHeight=100):
                        w.subset_info(subset_info)
        with c2:
            working_subset = st.selectbox(
                "Select a subset to apply NDR:", subsets, key="sb_select_subset_ndr_sin"
            )
            duplicated_subset = st.text_input(
                "Subset name for the removed data:", key="ti_select_subset_ndr_sin"
            )
            advanced_option = st.toggle("Advanced option", key="tg_advanced_opt_ndr_sin")
            if advanced_option:
                num_items = len(data_helper.dataset().get_subset(working_subset))
                num_cut = st.number_input(
                    "Maximum output dataset size:",
                    value=min(100, num_items),
                    step=1,
                    key="ni_num_cut_ndr_sin",
                )
                over_sample = st.radio(
                    "Oversample Policy",
                    ["random", "similarity"],
                    help="Specify the strategy when num_cut > length of the result after removal.",
                    captions=[
                        "Sample from removed data randomly",
                        "Select from removed data with ascending order of similarity",
                    ],
                    key="rd_over_sample_ndr_sin",
                )
                under_sample = st.radio(
                    "Undersample Policy",
                    ["uniform", "inverse"],
                    help="Specify the strategy when num_cut < length of the result after removal.",
                    captions=[
                        "Sample data with uniform distribution",
                        "Select data with reciprocal of the number",
                    ],
                    key="rd_under_sample_ndr_sin",
                )
            else:
                num_cut = None
                over_sample = None
                under_sample = None

            btn_find_nd = st.button(
                "Find Near Duplicate",
                use_container_width=True,
                key="btn_find_ndr_sin",
            )
            if btn_find_nd:
                self._run_ndr(
                    data_helper,
                    working_subset,
                    duplicated_subset,
                    num_cut,
                    over_sample,
                    under_sample,
                )
            try:
                duplicated = data_helper.dataset().get_subset(duplicated_subset)
            except KeyError:
                duplicated = []

            btn_remove_nd = st.button(
                "Remove Near Duplicate",
                use_container_width=True,
                disabled=len(duplicated) == 0,
                key="btn_remove_ndr_sin",
            )
            if btn_remove_nd:
                self._remove_duplicated(data_helper, duplicated_subset)

            if len(duplicated) > 0:
                with elements("single-transform-ndr-duplicated"):
                    board = Dashboard()
                    w = SimpleNamespace(
                        dashboard=board,
                        player=Gallery(board, 0, 0, 8, 3, minH=3),
                    )
                    with st.container():
                        with w.dashboard(rowHeight=100):
                            w.player(duplicated, max_number=12, title=duplicated_subset)


class TransformCategory(NamedTuple):
    type: str
    transforms: Tuple[TransformBase]


def on_click(transform: TransformBase):
    state["selected_transform"] = transform()


def main():
    check_image_backend(state.get("IMAGE_BACKEND"))

    log.info(f"{__file__} called")
    data_helper: SingleDatasetHelper = state["data_helper"]
    transform_categories = (
        TransformCategory("Category Management", (TransformLabelRemap,)),
        TransformCategory(
            "Subset Management",
            (
                TransformAggregation,
                TransformSplit,
                TransformSubsetRename,
            ),
        ),
        TransformCategory(
            "Item Management",
            (
                TransformReindexing,
                TransformFiltration,
                TransformRemove,
                TransformAutoCorrection,
                TransformNDR,
            ),
        ),
    )
    if "selected_transform" not in state or state["selected_transform"] is None:
        state["selected_transform"] = transform_categories[0].transforms[0]()

    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        st.subheader("Transforms")
        for category in transform_categories:
            sac.divider(label=category.type, icon="map", align="center")
            for transform in category.transforms:
                transform_name = transform().name
                st.button(
                    transform_name,
                    use_container_width=True,
                    on_click=on_click,
                    args=(transform,),
                    key="btn_" + "_".join(word.lower() for word in transform_name.split()),
                )
    with c2:
        transform = state["selected_transform"]
        st.subheader(transform.name)
        info_str = transform.info
        if transform.link != "":
            info_str += f" For more details, please refer to [this link]({transform.link})"
        st.info(info_str)

        transform.gui(data_helper)
