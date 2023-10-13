import json
import io
import os
import os.path as osp
import tempfile
import zipfile
import numpy as np
import streamlit as st
import time

from pathlib import Path
from types import SimpleNamespace
from matplotlib import pyplot as plt
from PIL import Image
from streamlit import session_state as state
from streamlit_elements import dashboard, elements, sync, event, mui, html
import streamlit_antd_components as sac

from datumaro.components.annotation import AnnotationType
from datumaro.components.dataset import Dataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT            
from datumaro.components.hl_ops import HLOps
from datumaro.components.visualizer import Visualizer
from datumaro.plugins.validators import ClassificationValidator, DetectionValidator, SegmentationValidator

from .dashboard import Bar, Dashboard, DataGrid, Radar, Pie, Gallery

def main():
    st.write(
        """
        :factory: Dataset management &nbsp; [![GitHub][github_badge]][github_link] [![PyPI][pypi_badge]][pypi_link]
        =====================

        Import a dataset and manipulate it!

        [github_badge]: https://badgen.net/badge/icon/GitHub?icon=github&color=black&label
        [github_link]: https://github.com/openvinotoolkit/datumaro

        [pypi_badge]: https://badgen.net/pypi/v/streamlit-elements?icon=pypi&color=black&label
        [pypi_link]: https://pypi.org/project/datumaro/
        """
    )

    # Define a custom CSS style
    custom_css = """
    <style>
        .css-q8sbsg p {
            font-size: 16px;
        }
        .container-outline p {
            border: 2px solid #000; /* Adjust the border properties as needed */
            padding: 10px; /* Adjust the padding as needed */
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


    keys = ['file_id', 'path', 'format', 'task', 'dataset', 'subset', 'validator']
    for k in keys:
        if k not in state:
            state[k] = None

    if state['subset'] is None:
        state['subset'] = 0

    temp_dir = tempfile.mkdtemp()
    with st.expander("Import a dataset"):
        uploaded_zip = st.file_uploader("Upload a zip file containing dataset", type=["zip"])

        if uploaded_zip is not None:
            if uploaded_zip != state['file_id']:
                # Extract the contents of the uploaded zip file to the temporary directory
                zip_path = osp.join(temp_dir, "uploaded_images.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.read())

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                print("temp_dir", temp_dir)

                # Find the path to the original folder
                orig_path = None
                for root, dirs, _ in os.walk(temp_dir):
                    if len(dirs) == 1:
                        orig_path = osp.join(root, dirs[0])
                        break
                    else:
                        orig_path = osp.join(root, "..")
                
                state['file_id'] = uploaded_zip
                state['path'] = orig_path

            # Display the list of image files in the UI
            detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(path=state['path'])
            selected_format = st.selectbox("Select a format to import:", detected_formats)

            if selected_format is not None:
                if selected_format != state['format']:
                    state['dataset'] = Dataset.import_from(path=state['path'], format=selected_format)
                    state['format'] = selected_format

                tasks = ["classification", "detection", "segmentation"]
                selected_task = st.selectbox("Select a task for validation:", tasks)

                if selected_task is not None:
                    if selected_task != state['task']:
                        if selected_task == "classification":
                            state['validator'] = ClassificationValidator()
                        elif selected_task == "detection":
                            state['validator'] = DetectionValidator()
                        elif selected_task == "segmentation":
                            state['validator'] = SegmentationValidator()
                        
                        state['task'] = selected_task

    st.title("")

    def call_general():
        dataset = state['dataset']
        with elements("general"):
            reports = state['validator'].validate(dataset)
            state['reports'] = reports

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
                
            val_info = {}
            anomaly_info = []
            for report in reports["validation_reports"]:
                anomaly_type = report["anomaly_type"]
                val_info[anomaly_type] = val_info.get(anomaly_type, 0) + 1
                
                anomaly_info.append(
                    {
                        "anomaly": anomaly_type,
                        "subset": report.get("subset", "None"),
                        "id": report.get("item_id", "None"),
                        "description": report["description"]
                    }
                )

            val_info_dict = []
            for type, cnt in val_info.items():
                temp_dict = {
                    "id": type,
                    "label": type,
                    "value": cnt,
                }
                val_info_dict.append(temp_dict)

            board = Dashboard()
            w = SimpleNamespace(
                dashboard=board,
                subset_info=Pie(name="Subset info", 
                                **{'board': board, 'x': 0, 'y': 0, 'w': 3, 'h': 6, 'minW': 2, 'minH': 4}),
                cat_info=Radar(name="Category info", indexBy="subset", keys=[cat.name for cat in categories.items],
                                **{'board': board, 'x': 3, 'y': 0, 'w': 3, 'h': 6, 'minW': 2, 'minH': 4}),
                val_info=Bar(name="Validation info", 
                                **{'board': board, 'x': 0, 'y': 7, 'w': 6, 'h': 6, 'minW': 2, 'minH': 4}),
                player=Gallery(board, 6, 0, 6, 12, minH=5),
                data_grid=DataGrid(board, 0, 12, 12, 6, minH=4),
            )

            with w.dashboard(rowHeight=50):
                w.subset_info(subset_info_dict)
                w.cat_info(cat_info_dict)
                w.val_info(val_info_dict)
                w.player(dataset)
                w.data_grid(data=anomaly_info, grid_name="Validation report")

    def call_visualize():
        dataset = state['dataset']
        with elements("visualize"):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader("Parameters")
                selected_subset = st.selectbox("Select a subset:", dataset.subsets())
                if selected_subset:
                    ids = [item.id for item in dataset.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a dataset item:", ids)

                if selected_id:
                    item = dataset.get(selected_id, selected_subset)
                    ann_ids = ["All",] + [ann.id for ann in item.annotations]
                    selected_ann_id = st.selectbox("Select a dataset item:", ann_ids)

                selected_alpha = st.select_slider(
                    "Choose a transparency of annotations",
                    options = np.arange(0.0, 1.1, 0.1, dtype=np.float16)
                )

                visualizer = Visualizer(dataset, figsize=(8, 8), alpha=selected_alpha)

            with c2:
                st.subheader("Item")
                if selected_ann_id == "All":
                    fig = visualizer.vis_one_sample(selected_id, selected_subset)
                else:
                    fig = visualizer.vis_one_sample(selected_id, selected_subset, ann_id=selected_ann_id)
                fig.set_facecolor('none')

                # Save the Matplotlib figure to a BytesIO buffer as PNG
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                plt.close(fig)
                buffer.seek(0)
                img = Image.open(buffer)

                st.image(img, use_column_width=True)

    def call_explore():
        pass

    def call_analyze():
        pass

    def call_transform():
        dataset = state['dataset']
        with elements("transform"):
            with st.expander("Category Management"):
                sac.divider(label='Label remapping', icon='map', align='center', bold=False)

            with st.expander("Subset Management"):
                sac.divider(label='Aggregation', icon='columns', align='center', bold=False)
                st.info("This helps to merge all subsets within a dataset into a single **default** subset.")
                aggre_subset_btn = st.button("Do aggregation")
                if aggre_subset_btn:
                    state['dataset'] = HLOps.aggregate(dataset, from_subsets=dataset.subsets(), to_subset="default")
                    st.toast('Success!', icon='🎉')
                    # success = st.success("Success!")
                    # time.sleep(1)
                    # success.empty()

                sac.divider(label='Split', icon='columns-gap', align='center', bold=False)
                st.info("This helps to divide a dataset into multiple subsets with a given ratio.")
                col1, col2  = st.columns(9)[:2]
                with col1:
                    add_subset_btn = st.button("Add subset", use_container_width=True)
                with col2:
                    split_btn = st.button("Do split", use_container_width=True)

                if add_subset_btn:
                    state['subset'] += 1
                
                name, ratio = st.columns(5)[:2]
                splits = []
                for idx in range(state['subset']):
                    with name:
                        subset_name = st.text_input(key=f"subset_name_{idx}", label="Enter subset name", value="train")
                    with ratio:
                        subset_ratio = st.text_input(key=f"subset_ratio_{idx}", label="Enter subset ratio", value=0.5)
                    splits.append((subset_name, float(subset_ratio)))
                
                if split_btn:
                    state['dataset'] = dataset.transform("random_split", splits=splits)
                    state['subset'] = 1
                    st.toast('Success!', icon='🎉')

            with st.expander("Item Management"):
                sac.divider(label='Reindexing', icon='stickies-fill', align='center', bold=False)
                st.info("This helps to reidentify all items.")
                col1, col2 = st.columns(6)[:2]
                with col1:
                    item_reindex_btn = st.button("Set IDs from 0", use_container_width=True)
                    
                with col2:
                    item_media_name_btn = st.button("Set IDs with media name", use_container_width=True)
                    
                if item_reindex_btn:
                    state['dataset'] = dataset.transform("reindex", start=0)
                    st.toast('Success!', icon='🎉')

                if item_media_name_btn:
                    state['dataset'] = dataset.transform("id_from_image_name")
                    st.toast('Success!', icon='🎉')
                
                sac.divider(label='Filtration', icon='funnel-fill', align='center', bold=False)
                sac.divider(label='Remove', icon='eraser-fill', align='center', bold=False)
                st.info("This helps to remove some items or annotations within a dataset.")
                subset, item, annotation = st.columns(5)[:3]
                
                with subset:
                    selected_subset = st.selectbox("Select a subset:", dataset.subsets())

                with item:
                    ids = [item.id for item in dataset.get_subset(selected_subset)]
                    selected_id = st.selectbox("Select a subset item:", ids)
                    
                with annotation:
                    item = dataset.get(selected_id, selected_subset)
                    ann_ids = ["All",] + [ann.id for ann in item.annotations]
                    selected_ann_id = st.selectbox("Select a item annotation:", ann_ids)

                col1, col2 = st.columns(6)[:2]
                with col1:
                    rm_item_btn = st.button("Remove item", use_container_width=True)

                with col2:
                    rm_ann_btn = st.button("Remove annotation", use_container_width=True)

                if rm_item_btn:
                    state['dataset'] = dataset.transform("remove_items", ids=[(selected_id, selected_subset)])
                    st.toast('Success!', icon='🎉')

                if rm_ann_btn:
                    if selected_ann_id == "All":
                        state['dataset'] = dataset.transform("remove_annotations", ids=[(selected_id, selected_subset)])
                    else:
                        state['dataset'] = dataset.transform("remove_annotations", ids=[(selected_id, selected_subset, selected_ann_id)])
                    st.toast('Success!', icon='🎉')

                sac.divider(label='Auto-correction', icon='hammer', align='center', bold=False)
                st.info("This helps to correct a dataset and clean up validation report.")

                col1, col2 = st.columns(6)[:2]
                with col1:
                    correct_btn = st.button("Correct a dataset", use_container_width=True)
                
                if correct_btn:
                    state['dataset'] = dataset.transform("correct", reports=state['reports'])
                    st.toast('Success!', icon='🎉')

                # sac.divider(label='SUBSET MAPPING', icon='code-square', align='center', bold=False)
                # with st.container():
                #     rename_subset_btn = st.button("Rename subsets")
                #     name1, emo, name2 = st.columns(8)[:3]                        
                #     for idx, subset in enumerate(state['dataset'].subsets()):
                #         with name1:
                #             st.write("")
                #             st.write("\n\n\n\n\n\n")
                #             st.write(str(subset))
                #         with emo:
                #             st.write("")
                #             st.write("\n\n\n\n\n\n")
                #             st.write(":arrow_right:")
                #         with name2:
                #             subset_new_name = st.text_input(key=f"subset_new_name_{idx}", label="New subset name", value="train")
                            
                #     if rename_subset_btn:
                #         state['dataset'] = dataset.transform("random_split", splits=splits)


            # with mui.Accordion:
            #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
            #         mui.Typography("Category Management", fontSize=16)
            #     with mui.AccordionDetails():
            #         with mui.Box(sx={"flex": 1, "borderTop": 1, "borderBottom": 1, "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}):
            #             mui.Typography(mui.icon.LooksOne, " Label remapping", fontSize=16)
            #             columns = [
            #                 { "field": 'id', "headerName": 'Label ID', "width": 100},
            #                 { "field": 'cur_name', "headerName": 'Label Name', "width": 200, "editable": False},
            #                 { "field": 'new_name', "headerName": 'New Label Name', "width": 200, "editable": True},
            #                 { "field": 'num_ann', "headerName": 'Number of annotations', "width": 200, "editable": False},
            #             ]
                        
            #             categories = state['dataset'].categories()[AnnotationType.label]
            #             labels = []
            #             for cat_name, idx in categories._indices.items():
            #                 labels.append({'id': idx, 'cur_name': cat_name, 'new_name': cat_name, 'num_ann': None})
                        
            #             board = Dashboard()
            #             w = SimpleNamespace(
            #                 dashboard=board,
            #                 data_grid=DataGrid(board, 0, 0, 8, 7, minH=len(labels)),
            #             )

            #             with w.dashboard(rowHeight=50):
            #                 w.data_grid(data=labels, grid_name="Label schema", columns=columns)

            #             mui.Button("Do label remapping", sx={'border': 1, 'color': 'white', 'background': 'black'})

            # with mui.Accordion:
            #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
            #         mui.Typography("Subset Management")
            #     with mui.AccordionDetails():
            #         with mui.Box(sx={"flex": 1, "borderTop": 1, "borderBottom": 1, "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}):
            #             mui.Typography(mui.icon.LooksOne, " Aggregation", fontSize=16, sx={"paddingBottom": 2})

            #         with mui.Box(sx={"flex": 1, "borderTop": 1, "borderBottom": 1, "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}):
            #             mui.Typography(mui.icon.LooksTwo, " Split", fontSize=16, sx={"paddingBottom": 2})

            #             with mui.ButtonGroup(sx={"aria-label": "split button"}):
            #                 add_subset_btn = mui.Button("Add subset", sx={'border': 1, 'color': 'white', 'background': 'black'})
            #                 split_subset_btn = mui.Button("Do split", sx={'border': 1, 'color': 'white', 'background': 'black'})
                        
            #             if add_subset_btn:
            #                 state['subset'] += 1
                        
            #             splits = []
            #             for idx in range(state['subset']):
            #                 mui.Typography("")
            #                 subset_name = mui.TextField(id=f"name_{idx}", label="subset name", defaultValue="default", variant="outlined", size="small")
            #                 subset_ratio = mui.TextField(id=f"ratio_{idx}", label="subset ratio", defaultValue="1.0", variant="outlined", size="small")
            #                 print(subset_name, subset_ratio)
            #                 splits.append((subset_name, float(subset_ratio)))
    
            #             if split_subset_btn:
            #                 state['dataset'] = dataset.transform("random_split", splits=splits)
            #                 print(state['dataset'])
            #                 state['subset'] = 0

            #                 success = st.success("Success!")
            #                 time.sleep(1)
            #                 success.empty()

            # with mui.Accordion:
            #     with mui.AccordionSummary(expandIcon=mui.icon.ExpandMore, sx={'background': 'grey[900]', 'border': 1}):
            #         mui.Typography("Item Management", fontSize=16) #, fontFamily='Helvetica Neue'
            #     with mui.AccordionDetails():
            #         with mui.Box(sx={"flex": 1, "borderTop": 1, "borderBottom": 1, "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}):
            #             mui.Typography(mui.icon.LooksOne, " ID reindexing ", fontSize=16, sx={"paddingBottom": 1})
            #             with mui.ButtonGroup(sx={"aria-label": "split button"}):
            #                 mui.Button("Set IDs from 0", sx={'border': 1, 'color': 'white', 'background': 'black'})
            #                 mui.Button("Set IDs from media name", sx={'border': 1, 'color': 'white', 'background': 'black'})

            #         with mui.Box(sx={"flex": 1, "borderTop": 1, "borderBottom": 1, "borderColor": "divider", "paddingTop": 1, "paddingBottom": 2}):
            #             mui.Typography("Filtration")

    def call_export():
        tasks = ["classification", "detection", "instance_segmentation", "segmentation", "landmark"]
        formats = {
            "classification": ["imagenet", "cifar", "mnist", "mnist_csv", "lfw"],
            "detection": ["coco_instances", "voc_detection", "yolo", "yolo_ultralytics", "kitti_detection", "tf_detection_api", "open_images", "segment_anything", "mot_seq_gt", "wider_face"],
            "instance_segmentation": ["coco_instances", "voc_instance_segmentation", "open_images", "segment_anything"],
            "segmentation": ["coco_panoptic", "voc_segmentation", "kitti_segmentation", "cityscapes", "camvid"],
            "landmark": ["coco_person_keypoints", "voc_layout", "lfw"],
        }
        selected_task = st.selectbox("Select a task to export:", tasks)
        if selected_task:
            selected_format = st.selectbox("Select a format to export:", formats[selected_task])
        
        if selected_task and selected_format:
            selected_path = st.text_input("Select a path to export:", value=osp.join(osp.expanduser("~"), "Downloads"))

        export_btn = st.button("Export")
        if export_btn:
            if not osp.exists(selected_path):
                os.makedirs(selected_path)
            print(osp.abspath(selected_path))
            state['dataset'].export(selected_path, format=selected_format, save_media=True)

    if state['dataset'] is not None:
        select_tab = sac.tabs([
            sac.TabsItem(label='GENERAL', icon='incognito'),
            sac.TabsItem(label='VISUALIZE', icon='image'),
            sac.TabsItem(label='EXPLORE', icon='tags', disabled=True),
            sac.TabsItem(label='ANALYZE', icon='clipboard2-data-fill', disabled=True),
            sac.TabsItem(label='TRANSFORM', icon='tools'),
            sac.TabsItem(label='EXPORT', icon='cloud-arrow-down'),
        ], format_func='title', align='center')

        if select_tab == "GENERAL":
            call_general()

        if select_tab == "TRANSFORM":
            call_transform()
            
        if select_tab == "VISUALIZE":
            call_visualize()
            
        if select_tab == "EXPLORE":
            call_explore()

        if select_tab == "ANALYZE":
            call_analyze()

        if select_tab == "EXPORT":
            call_export()

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
