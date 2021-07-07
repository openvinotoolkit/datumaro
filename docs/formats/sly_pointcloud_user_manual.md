# Supervisely Point Cloud user manual

## Contents

- [Format specification](#format-specification)
- [Load Point Cloud dataset](#load-point-cloud-dataset)

## Format specification

Point Cloud data format specification available [here](https://docs.supervise.ly/data-organization/00_ann_format_navi). An example is [here](https://drive.google.com/file/d/1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl/view).

##  Load Point Cloud dataset

The point cloud dataset is available for download:

https://drive.google.com/u/0/uc?id=1BtZyffWtWNR-mk_PHNPMnGgSlAkkQpBl&export=download


Point Cloud dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── ds0/
    │   ├── ann/
    │   │   ├── <pcdname1.pcd.json>
    │   │   ├── <pcdname2.pcd.json>
    │   │   └── ...
    │   ├── pointcloud/
    │   │   ├── <pcdname1.pcd>
    │   │   ├── <pcdname1.pcd>
    │   │   └── ...
    │   ├── related_images/
    │   │   ├── <pcdname1_pcd>/
    │   │   |  ├── <image_name.ext.json>
    │   │   |  ├── <image_name.ext.json>
    │   │   └── ...
    ├── key_id_map.json
    ├── meta.json
```
