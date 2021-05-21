# Velodyne Points user manual

## Contents

- [Format specification](#format-specification)
- [Velodyne Points dataset](#load-velodyne-points-dataset)

## Format specification

Velodyne Points/Kitti data format specification available [here](http://www.cvlibs.net/datasets/kitti/index.php).

##  Load Velodyne Points dataset

The velodyne points/Kitti dataset is available for download:

https://cloud.enterprise.deepsystems.io/s/YcyfIf5zrS7NZcI/download

Kitti dataset directory should have the following structure:

<!--lint disable fenced-code-flag-->
```
└─ Dataset/
    ├── image_00/
    │   ├── data/
    │   │   ├── <name1.ext>
    ├── image_01/
    │   ├── data/
    │   │   ├── <name2.ext>
    ├── tracklets.xml
    ├── velodyne_points/
    │   ├── data/
    │   │   ├── <name1.pcd>
    │   │   ├── <name2.pcd]>
```

The velodyne_points folder can contain files in pcd/bin format.