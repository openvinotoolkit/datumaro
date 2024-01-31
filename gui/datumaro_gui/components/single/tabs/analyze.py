# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import time
from collections import defaultdict
from types import SimpleNamespace

import streamlit as st
from datumaro_gui.utils.dataset.data_loader import SingleDatasetHelper
from datumaro_gui.utils.drawing import Chart, ChartWithTab, Dashboard, DataGrid, DatasetInfoBox
from streamlit import session_state as state
from streamlit_elements import elements


@st.cache_data
def get_dataset_info(stats_image, stats_anns, image_mean, n_labels):
    dataset_info = {}
    dataset_info["n_images"] = stats_image["dataset"]["images count"]
    dataset_info["n_unique"] = stats_image["dataset"]["unique images count"]
    dataset_info["n_repeated"] = stats_image["dataset"]["repeated images count"]
    dataset_info["avg_w"] = image_mean[1]
    dataset_info["avg_h"] = image_mean[0]

    dataset_info["n_subsets"] = len(stats_image["subsets"])

    dataset_info["n_anns"] = stats_anns["annotations count"]
    dataset_info["n_unannotated"] = stats_anns["unannotated images count"]
    dataset_info["n_labels"] = n_labels

    return dataset_info


@st.cache_data
def get_num_images_by_subset(stats_image):
    num_images_by_subset = []
    for subset, info in stats_image["subsets"].items():
        num_images_by_subset.append({"id": subset, "label": subset, "value": info["images count"]})
    return num_images_by_subset


COLS_REPEATED = [
    {"field": "repeated", "headerName": "Repeated Items(Subset-ID)", "flex": 1},
]
COLS_ITEM = [
    {"field": "item_id", "headerName": "ID", "flex": 1},
]
COLS_SUBSET_ITEM = [
    {"field": "subset", "headerName": "Subset"},
    {"field": "item_id", "headerName": "ID", "flex": 1},
]
COLS_NEGATIVE_INVALID = [
    {"field": "subset", "headerName": "Subset"},
    {"field": "item_id", "headerName": "Item ID", "flex": 1},
    {"field": "ann_id", "headerName": "Annotation ID", "flex": 1},
    {"field": "values", "headerName": "Values", "flex": 1},
]


@st.cache_data
def get_image_size_dist(image_size_info):
    def get_scatter_data(size_data):
        data = []
        total = 0
        for id, sizes in size_data.items():
            data.append({"id": id, "data": sizes})
            total += len(sizes)
        return data, total

    chart_kwargs = {
        "axisBottom": {
            "tickSize": 5,
            "legend": "Width",
            "legendOffset": 40,
        },
        "axisLeft": {
            "tickSize": 5,
            "legend": "Height",
            "legendOffset": -45,
        },
        # "blendMode": "soft-light",
    }

    tab_data = []
    start_time = time.time()
    subset_data, subset_total = get_scatter_data(image_size_info["by_subsets"])
    tab_data.append(
        {
            "title": f"By Subsets ({subset_total} Images)",
            "data": subset_data,
            "chart_type": Dashboard.Chart.ScatterPlot,
            "chart_kwargs": chart_kwargs,
        }
    )
    print("get_scatter_data by subsets time : ", time.time() - start_time)
    start_time = time.time()
    label_data, label_total = get_scatter_data(image_size_info["by_labels"])
    print("get_scatter_data by labels time : ", time.time() - start_time)
    start_time = time.time()
    tab_data.append(
        {
            "title": f"By Labels ({label_total} Labels)",
            "data": label_data,
            "chart_type": Dashboard.Chart.ScatterPlot,
            "chart_kwargs": chart_kwargs,
        }
    )
    print("tab data append time : ", time.time() - start_time)

    if subset_total == 0 and label_total == 0:
        tab_data = None

    return tab_data


@st.cache_data
def get_repeated_images(stats_image):
    grid_data = []
    for i, items in enumerate(stats_image["dataset"]["repeated images"]):
        grid_data.append(
            {"id": i, "repeated": ", ".join([f"{subset}-{id}" for id, subset in items])}
        )
    return grid_data


@st.cache_data
def get_unannotated_images(stats_anns):
    grid_data = []
    # print("unannotated:", stats_anns["unannotated images"])
    for i, id in enumerate(stats_anns["unannotated images"]):
        grid_data.append({"id": i, "item_id": id})
    return grid_data


@st.cache_data
def get_num_anns_by_type(stats_anns):
    num_anns_by_type = []
    for key, val in stats_anns["annotations by type"].items():
        if val["count"] > 0:
            num_anns_by_type.append({"id": key, "label": key, "value": val["count"]})
    return num_anns_by_type


@st.cache_data
def get_label_dist(stats_anns):
    labels = stats_anns["annotations"]["labels"]
    label_dist_info = []
    for key, val in labels["distribution"].items():
        label_dist_info.append({"id": key, "label": key, "value": val[0]})
    return label_dist_info


@st.cache_data
def get_attr_dist(stats_anns):
    attributes = stats_anns["annotations"]["labels"]["attributes"]
    attr_dist_info = []
    attr_dist_total = 0
    for attr, info in attributes.items():
        data = []
        for key in info["values present"]:
            count = info["distribution"][key][0]
            data.append({"id": key, "label": key, "value": count})
            attr_dist_total += count
        tab_info = {"title": attr, "data": data, "chart_type": Dashboard.Chart.Pie}
        attr_dist_info.append(tab_info)

    if attr_dist_total == 0:
        return None

    return attr_dist_info


@st.cache_data
def get_segments_dist(stats_anns):
    segments = stats_anns["annotations"]["segments"]
    seg_dist_info = []

    area_dist = []
    area_dist_total = 0
    for item in segments["area distribution"]:
        key = f"{item['min']:,.2f}~{item['max']:,.2f}"
        area_dist.append({"id": key, "label": key, "value": item["count"]})
        area_dist_total += item["count"]
    seg_dist_info.append(
        {"title": "Bbox Area Distribution", "data": area_dist, "chart_type": Dashboard.Chart.Bar}
    )
    pixel_dist = []
    pixel_dist_total = 0
    for key, dist in segments["pixel distribution"].items():
        pixel_dist.append({"id": key, "label": key, "value": dist[0]})
        pixel_dist_total += dist[0]
    seg_dist_info.append(
        {"title": "Pixel Distribution", "data": pixel_dist, "chart_type": Dashboard.Chart.Pie}
    )
    if area_dist_total == 0 and pixel_dist_total == 0:
        return None

    return seg_dist_info


@st.cache_data
def get_tab_data_for_val_label_dist(val_report):
    tab_data = []

    def calc_piechart_data(label_dist):
        data = []
        total = 0
        # print(label_dist)
        for label, count in label_dist.items():
            data.append({"id": label, "label": label, "value": count})
            total += count
        if len(data) == 0:
            data = None
        return data, total

    label_distribution = val_report["statistics"]["label_distribution"]
    data, total = calc_piechart_data(label_distribution["defined_labels"])
    tab_data.append(
        {"title": f"Defined Labels ({total:,})", "data": data, "chart_type": Dashboard.Chart.Pie}
    )

    data, total = calc_piechart_data(label_distribution["undefined_labels"])
    tab_data.append(
        {"title": f"Undefined Labels({total:,})", "data": data, "chart_type": Dashboard.Chart.Pie}
    )

    return tab_data


@st.cache_data
def get_tab_data_for_val_attr_dist(val_report):
    tab_data = []

    def calc_tab_data(attr_dist):
        children = []
        total = 0
        for label, attrs in attr_dist.items():
            for attr, attr_info in attrs.items():
                value_dist = []
                for value, count in attr_info["distribution"].items():
                    value_dist.append({"id": f"{label}/{attr}-{value}", "value": count})
                    total += count
                if len(value_dist) > 0:
                    children.append({"id": f"{label}/{attr}", "children": value_dist})
        if len(children) == 0:
            data = None
        else:
            data = {"id": "nivo", "children": children}
        return data, total

    attr_distribution = val_report["statistics"]["attribute_distribution"]
    data_defined, total = calc_tab_data(attr_distribution["defined_attributes"])
    tab_data.append(
        {
            "title": f"Defined Attributes ({total:,})",
            "data": data_defined,
            "chart_type": Dashboard.Chart.Sunburst,
            "chart_kwargs": {
                "enableArcLabels": True,
                "childColor": {
                    "from": "color",
                    "modifiers": [
                        ["brighter", "0.5"],
                    ],
                },
            },
        }
    )

    data_undefined, total = calc_tab_data(attr_distribution["undefined_attributes"])
    tab_data.append(
        {
            "title": f"Undefined Attributes ({total:,})",
            "data": data_undefined,
            "chart_type": Dashboard.Chart.Sunburst,
        }
    )

    if data_defined is None and data_undefined is None:
        tab_data = None

    return tab_data


@st.cache_data
def get_grid_data_val_missing_annotations(val_report):
    grid_data = []
    for i, (id, subset) in enumerate(val_report["statistics"]["items_missing_annotation"]):
        grid_data.append(
            {
                "id": i,
                "item_id": id,
                "subset": subset,
            }
        )
    return grid_data


@st.cache_data
def get_grid_data_val_negative_length(val_report):
    grid_data = []
    index = 0
    for (id, subset), anns in val_report["statistics"]["items_with_negative_length"].items():
        for ann_id, values in anns.items():
            grid_data.append(
                {"id": index, "subset": subset, "item_id": id, "ann_id": ann_id, "values": values}
            )
            index += 1
    return grid_data


@st.cache_data
def get_grid_data_val_invalid_value(val_report):
    grid_data = []
    index = 0
    for (id, subset), anns in val_report["statistics"]["items_with_invalid_value"].items():
        for ann_id, values in anns.items():
            grid_data.append(
                {"id": index, "subset": subset, "item_id": id, "ann_id": ann_id, "values": values}
            )
            index += 1
    return grid_data


@st.cache_data
def get_grid_data_val_multiple_labels(val_report):
    grid_data = []
    for i, (id, subset) in enumerate(val_report["statistics"]["items_with_multiple_labels"]):
        grid_data.append(
            {
                "id": i,
                "item_id": id,
                "subset": subset,
            }
        )

    return grid_data


@st.cache_data
def get_validation_summary(val_report):
    if len(val_report["validation_reports"]) == 0:
        return None

    val_info = {}
    for report in val_report["validation_reports"]:
        anomaly_type = report["anomaly_type"]
        val_info[anomaly_type] = val_info.get(anomaly_type, 0) + 1

    bar_data = []
    for type, count in val_info.items():
        bar_data.append({"id": type, "label": type, "value": count})

    return bar_data


@st.cache_data
def get_anomaly_info(val_report):
    if len(val_report["validation_reports"]) == 0:
        return None

    anomaly_info = []
    for idx, report in enumerate(val_report["validation_reports"]):
        anomaly_info.append(
            {
                "id": idx,
                "anomaly": report["anomaly_type"],
                "subset": report.get("subset", "None"),
                "item_id": report.get("item_id", "None"),
                "description": report["description"],
            }
        )

    return anomaly_info  # for data_frid


@st.cache_data
def get_tab_data_for_label_dist_by_type(val_cls, val_det, val_seg):
    tab_data_defined = []
    tab_data_undefined = []

    (
        data_defined_label,
        key_defined_label,
        data_undefined_label,
        key_undefined_label,
    ) = get_radar_data_for_label_dist_by_type(val_cls, val_det, val_seg)

    if data_defined_label:
        tab_data_defined.append(
            {
                "title": "All",
                "data": data_defined_label,
                "chart_type": Dashboard.Chart.Radar,
                "chart_kwargs": {
                    "indexBy": "id",
                    "keys": key_defined_label,
                },
            }
        )

    if data_undefined_label:
        tab_data_undefined.append(
            {
                "title": "All",
                "data": data_undefined_label,
                "chart_type": Dashboard.Chart.Radar,
                "chart_kwargs": {
                    "indexBy": "id",
                    "keys": key_undefined_label,
                },
            }
        )

    if val_cls:
        defined, undefined = get_tab_data_for_val_label_dist(val_cls)
        if defined["data"]:
            tab_data_defined.append(
                {"title": "Label", "data": defined["data"], "chart_type": Dashboard.Chart.Pie}
            )
        if undefined["data"]:
            tab_data_undefined.append(
                {"title": "Label", "data": undefined["data"], "chart_type": Dashboard.Chart.Pie}
            )
    if val_det:
        defined, undefined = get_tab_data_for_val_label_dist(val_det)
        if defined["data"]:
            tab_data_defined.append(
                {"title": "Bbox", "data": defined["data"], "chart_type": Dashboard.Chart.Pie}
            )
        if undefined["data"]:
            tab_data_undefined.append(
                {"title": "Bbox", "data": undefined["data"], "chart_type": Dashboard.Chart.Pie}
            )
    if val_seg:
        defined, undefined = get_tab_data_for_val_label_dist(val_seg)
        if defined["data"]:
            tab_data_defined.append(
                {"title": "Polygon", "data": defined["data"], "chart_type": Dashboard.Chart.Pie}
            )
        if undefined["data"]:
            tab_data_undefined.append(
                {"title": "Polygon", "data": undefined["data"], "chart_type": Dashboard.Chart.Pie}
            )
    if not tab_data_defined:
        tab_data_defined = None
    if not tab_data_undefined:
        tab_data_undefined = None

    return tab_data_defined, tab_data_undefined


@st.cache_data
def get_radar_data_for_label_dist_by_type(val_cls, val_det, val_seg):
    defined_labels = defaultdict(dict)
    undefined_labels = defaultdict(dict)

    def calc_bar_data(val_report, ann_type):
        label_dist = val_report["statistics"]["label_distribution"]
        for label, count in label_dist["defined_labels"].items():
            defined_labels[label][ann_type] = count
        for label, count in label_dist["undefined_labels"].items():
            undefined_labels[label][ann_type] = count

    if val_cls:
        calc_bar_data(val_cls, "label")
    if val_det:
        calc_bar_data(val_det, "bbox")
    if val_seg:
        calc_bar_data(val_seg, "polygon")

    bar_data_defined = []
    bar_data_undefined = []
    keys_defined = set()
    keys_undefined = set()
    for id, counts in defined_labels.items():
        keys_defined.update(counts.keys())
        counts["id"] = id
        bar_data_defined.append(counts)

    for id, counts in undefined_labels.items():
        keys_undefined.update(counts.keys())
        counts["id"] = id
        bar_data_undefined.append(counts)

    if len(bar_data_defined) == 0:
        bar_data_defined = None
    if len(bar_data_undefined) == 0:
        bar_data_undefined = None

    return bar_data_defined, keys_defined, bar_data_undefined, keys_undefined


@st.cache_data
def get_tab_data_for_attr_dist_by_type(val_cls, val_det, val_seg):
    tab_data_defined = []
    tab_data_undefined = []

    def get_tab_data(title, data):
        return {
            "title": title,
            "data": data,
            "chart_type": Dashboard.Chart.Sunburst,
            "chart_kwargs": {
                "enableArcLabels": True,
                "childColor": {
                    "from": "color",
                    "modifiers": [
                        ["brighter", "0.5"],
                    ],
                },
            },
        }

    def add_tab_data(val_report, ann_type):
        tab_data = get_tab_data_for_val_attr_dist(val_report)
        if tab_data:
            for data in tab_data:
                if data["data"] is not None:
                    tab = get_tab_data(ann_type, data["data"])
                    if data["title"].startswith("Defined"):
                        tab_data_defined.append(tab)
                    else:
                        tab_data_undefined.append(tab)

    if val_cls:
        add_tab_data(val_cls, "Label")
    if val_det:
        add_tab_data(val_det, "Bbox")
    if val_seg:
        add_tab_data(val_seg, "Polygon")

    if not tab_data_defined:
        tab_data_defined = None
    if not tab_data_undefined:
        tab_data_undefined = None

    return tab_data_defined, tab_data_undefined


@st.cache_data
def get_tab_data_point_dist_in_label(val_report):
    tabs = []
    keys = ["mean", "median"]
    by_targets = defaultdict(dict)
    for label, points in val_report["statistics"]["point_distribution_in_label"].items():
        for target, numerical_info in points.items():
            if target == "area(wxh)":
                target = "area"
            elif target == "ratio(w/h)":
                target = "ratio"
            if target in ["short", "long"]:
                continue  # skip
            by_targets[target][label] = {}
            for key in keys:
                by_targets[target][label][key] = numerical_info[key]

    for target, by_labels in by_targets.items():
        data = []
        for label, values in by_labels.items():
            values["id"] = label
            data.append(values)

        tabs.append(
            {
                "title": target,
                "data": data,
                "chart_type": Dashboard.Chart.Bar,
                "chart_kwargs": {
                    "indexBy": "id",
                    "keys": keys,
                    "groupMode": "grouped",
                    "valueFormat": ">-,.2f",
                },
            }
        )

    return tabs


def main():
    data_helper: SingleDatasetHelper = state["data_helper"]
    n_labels = data_helper.num_labels

    stats_image = data_helper.get_image_stats()  # state["stats_image"]
    stats_anns = data_helper.get_ann_stats()  # state["stats_anns"]
    image_size_info = data_helper.get_image_size_info()  # state["image_size_info"]

    cls_summary = state["cls_summary"]
    cls_anomaly_info = state["cls_anomaly_info"]
    det_summary = state["det_summary"]
    det_anomaly_info = state["det_anomaly_info"]
    seg_summary = state["seg_summary"]
    seg_anomaly_info = state["seg_anomaly_info"]

    defined_label = state["defined_label"]
    undefined_label = state["undefined_label"]
    defined_attr = state["defined_attr"]
    undefined_attr = state["undefined_attr"]

    # if stats_image is None:
    #     stats_image = data_helper.get_image_stats()
    #     state["stats_image"] = stats_image
    # if stats_anns is None:
    #     stats_anns = data_helper.get_ann_stats()
    #     state["stats_anns"] = stats_anns
    # if image_size_info is None:
    #     image_size_info = data_helper.get_image_size_info()
    #     state["image_size_info"] = image_size_info
    image_mean = image_size_info["image_size"]["mean"]
    start_time = time.time()
    size_tabs = get_image_size_dist(image_size_info)
    print("get_image_size_dist time : ", time.time() - start_time)

    start_time = time.time()
    num_images_by_subset = get_num_images_by_subset(stats_image)
    print("get_num_images_by_subset time : ", time.time() - start_time)
    start_time = time.time()
    num_anns_by_type = get_num_anns_by_type(stats_anns)
    print("get_num_anns_by_type time : ", time.time() - start_time)
    start_time = time.time()
    label_dist_info = get_label_dist(stats_anns)
    print("get_label_dist time : ", time.time() - start_time)

    start_time = time.time()
    attr_dist_info = get_attr_dist(stats_anns)
    print("get_attr_dist time : ", time.time() - start_time)

    anns_by_type = stats_anns["annotations by type"]

    start_time = time.time()
    val_cls = data_helper.validate("classification") if anns_by_type["label"]["count"] > 0 else None
    print("cls validate time : ", time.time() - start_time)
    start_time = time.time()
    val_det = data_helper.validate("detection") if anns_by_type["bbox"]["count"] > 0 else None
    print("det validate time : ", time.time() - start_time)
    start_time = time.time()
    val_seg = data_helper.validate("segmentation") if anns_by_type["polygon"]["count"] > 0 else None
    print("seg validate time : ", time.time() - start_time)

    if val_cls and cls_summary is None and cls_anomaly_info is None:
        cls_summary = get_validation_summary(val_cls)
        cls_anomaly_info = get_anomaly_info(val_cls)
        state["cls_summary"] = cls_summary
        state["cls_anomaly_info"] = cls_anomaly_info
    if val_det and det_summary is None and det_anomaly_info is None:
        det_summary = get_validation_summary(val_det)
        det_anomaly_info = get_anomaly_info(val_det)
        state["det_summary"] = det_summary
        state["det_anomaly_info"] = det_anomaly_info
    if val_seg and seg_summary is None and seg_anomaly_info is None:
        seg_summary = get_validation_summary(val_seg)
        seg_anomaly_info = get_anomaly_info(val_seg)
        state["seg_summary"] = seg_summary
        state["seg_anomaly_info"] = seg_anomaly_info

    if defined_label is None and undefined_label is None:
        defined_label, undefined_label = get_tab_data_for_label_dist_by_type(
            val_cls, val_det, val_seg
        )
        state["defined_label"] = defined_label
        state["undefined_label"] = undefined_label
    if defined_attr is None and undefined_label is None:
        defined_attr, undefined_attr = get_tab_data_for_attr_dist_by_type(val_cls, val_det, val_seg)
        state["defined_attr"] = defined_attr
        state["undefined_attr"] = undefined_attr

    with elements("analyze"):
        w = SimpleNamespace()

        ##################################
        ## dataset statistics
        ##################################
        w.board_dataset = Dashboard()
        global x_pos, y_pos

        def proceed_pos(w=4, h=3):
            global x_pos, y_pos
            x_pos += w
            if x_pos >= 12:
                x_pos = 0
                y_pos += h

        def get_board_kwargs(board, w=4, h=3, minw=3, minh=3):
            global x_pos, y_pos
            kwargs = {
                "board": board,
                "x": x_pos,
                "y": y_pos,
                "w": w,
                "h": h,
                "minW": minw,
                "minH": minh,
            }
            proceed_pos(w=w, h=h)
            return kwargs

        x_pos = 0
        y_pos = 0
        w.dataset_info = DatasetInfoBox(**get_board_kwargs(w.board_dataset))
        w.num_images = Chart(Dashboard.Chart.Pie, **get_board_kwargs(w.board_dataset))
        w.num_anns = Chart(Dashboard.Chart.Pie, **get_board_kwargs(w.board_dataset))

        with w.board_dataset("Dataset Statistics"):
            w.dataset_info(
                "Dataset Information",
                get_dataset_info(stats_image, stats_anns, image_mean, n_labels),
            )
            w.num_images("Images by Subsets", num_images_by_subset, icon=Dashboard.Icon.Collections)
            w.num_anns("Annotations by Types", num_anns_by_type, icon=Dashboard.Icon.Label)

        ##################################
        ## Media statistics
        ##################################
        w.board_media = Dashboard()
        x_pos = 0
        y_pos = 0

        repeated_images = get_repeated_images(stats_image)
        unannotated_images = get_unannotated_images(stats_anns)

        w.size_info = ChartWithTab(
            icon=Dashboard.Icon.ScatterPlot,
            title="Image Size Distribution",
            tabs=size_tabs,
            tab_state_key="analyze.image.size",
            **get_board_kwargs(w.board_media),
        )
        if repeated_images:
            w.repeated_images = DataGrid(**get_board_kwargs(w.board_media))
        if unannotated_images:
            w.unannotated_images = DataGrid(**get_board_kwargs(w.board_media))

        with w.board_media("Media Statistics"):
            w.size_info()
            if repeated_images:
                w.repeated_images(
                    data=repeated_images,
                    grid_icon=Dashboard.Icon.Warning,
                    grid_name="Repeated Images",
                    columns=COLS_REPEATED,
                )
            if unannotated_images:
                w.unannotated_images(
                    data=unannotated_images,
                    grid_icon=Dashboard.Icon.Warning,
                    grid_name="Unannotated Images",
                    columns=COLS_ITEM,
                )

        ##################################
        ## Annotation Statistics
        ##################################
        w.board_anns = Dashboard()
        x_pos = 0
        y_pos = 0
        if label_dist_info:
            w.label_dist = Chart(Dashboard.Chart.Pie, **get_board_kwargs(w.board_anns))
        if defined_label:
            w.defined_label_dist = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Defined Labels",
                tabs=defined_label,
                tab_state_key="analyze.defined.label.distribution",
                **get_board_kwargs(w.board_anns),
            )
        if undefined_label:
            w.undefined_label_dist = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Undefined Labels",
                tabs=undefined_label,
                tab_state_key="analyze.undefined.label.distribution",
                **get_board_kwargs(w.board_anns),
            )

        if attr_dist_info:
            w.attr_dist = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Attributes Distribution",
                tabs=attr_dist_info,
                tab_state_key="analyze.attributes",
                **get_board_kwargs(w.board_anns),
            )
        if defined_attr:
            w.defined_attr_dist = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Defined Attributes",
                tabs=defined_attr,
                tab_state_key="analyze.defined.attr.distribution",
                **get_board_kwargs(w.board_anns),
            )
        if undefined_attr:
            w.undefined_attr_dist = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Undefined Attributes",
                tabs=undefined_attr,
                tab_state_key="analyze.undefined.attr.distribution",
                **get_board_kwargs(w.board_anns),
            )

        if val_det:
            w.point_dist_bbox = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Bbox Statistics",
                tabs=get_tab_data_point_dist_in_label(val_det),
                tab_state_key="analyze.bbox.stats",
                **get_board_kwargs(w.board_anns),
            )
        if val_seg:
            w.point_dist_polygon = ChartWithTab(
                icon=Dashboard.Icon.Label,
                title="Polygon Statistics",
                tabs=get_tab_data_point_dist_in_label(val_seg),
                tab_state_key="analyze.polygon.stats",
                **get_board_kwargs(w.board_anns),
            )

        with w.board_anns("Annotations Statistics"):
            if label_dist_info:
                w.label_dist("Labels Distribution", label_dist_info, legends=False)
            if defined_label:
                w.defined_label_dist(legends=False)
            if undefined_label:
                w.undefined_label_dist()
            if attr_dist_info:
                w.attr_dist()
            if defined_attr:
                w.defined_attr_dist()
            if undefined_attr:
                w.undefined_attr_dist()
            if val_det:
                w.point_dist_bbox()
            if val_seg:
                w.point_dist_polygon()

        ##################################
        ## Validation on Classification
        ##################################
        if val_cls:
            x_pos = 0
            y_pos = 0
            w.board_val_cls = Dashboard()

            w.cls_summary = Chart(Dashboard.Chart.Bar, **get_board_kwargs(w.board_val_cls))
            w.cls_anomaly = DataGrid(**get_board_kwargs(w.board_val_cls, w=8))
            # w.cls_missing_anns = DataGrid(**get_board_kwargs(w.board_val_cls))
            # w.cls_multiple_labels = DataGrid(**get_board_kwargs(w.board_val_cls))

            with w.board_val_cls("Validation Results on Classification"):
                w.cls_summary("Summary", cls_summary, legends=False)
                w.cls_anomaly(cls_anomaly_info, grid_name="Anomaly Reports")

        ##################################
        ## Validation on Detection
        ##################################
        if val_det:
            x_pos = 0
            y_pos = 0
            w.board_val_det = Dashboard()
            w.det_summary = Chart(Dashboard.Chart.Bar, **get_board_kwargs(w.board_val_det))
            w.det_anomaly = DataGrid(**get_board_kwargs(w.board_val_det, w=8))

            with w.board_val_det("Validation Results on Detection"):
                w.det_summary("Summary", det_summary, legends=False)
                w.det_anomaly(det_anomaly_info, grid_name="Anomaly Reports")

        ##################################
        ## Validation on Segmentation
        ##################################
        if val_seg:
            x_pos = 0
            y_pos = 0
            w.board_val_seg = Dashboard()
            w.seg_summary = Chart(Dashboard.Chart.Bar, **get_board_kwargs(w.board_val_seg))
            w.seg_anomaly = DataGrid(**get_board_kwargs(w.board_val_seg, w=8))

            with w.board_val_seg("Validation Results on Segmentation"):
                w.seg_summary("Summary", seg_summary, legends=False)
                w.seg_anomaly(seg_anomaly_info, grid_name="Anomaly Reports")
