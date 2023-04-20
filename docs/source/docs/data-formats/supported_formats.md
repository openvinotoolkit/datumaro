# Supported Data Formats

List of supported formats:
- ADE20k (v2017) (import-only)
  - [Format specification](https://www.kaggle.com/soumikrakshit/ade20k)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/ade20k2017_dataset)
  - [Format documentation](../explanation/formats/ade20k2017)
- ADE20k (v2020) (import-only)
  - [Format specification](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/ade20k2020_dataset)
  - [Format documentation](../explanation/formats/ade20k2020)
- Align CelebA (`classification`, `landmarks`) (import-only)
  - [Format specification](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/align_celeba_dataset)
  - [Format documentation](../explanation/formats/align_celeba)
- BraTS (`segmentation`) (import-only)
  - [Format specification](https://www.med.upenn.edu/sbia/brats2018/data.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/brats_dataset)
  - [Format documentation](../explanation/formats/brats)
- BraTS Numpy (`detection`, `segmentation`) (import-only)
  - [Format specification](https://www.med.upenn.edu/sbia/brats2018/data.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/brats_numpy_dataset)
  - [Format documentation](../explanation/formats/brats_numpy)
- CamVid (`segmentation`)
  - [Format specification](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/camvid_dataset)
- CelebA (`classification`, `detection`, `landmarks`) (import-only)
  - [Format specification](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/celeba_dataset)
  - [Format documentation](../explanation/formats/celeba)
- CIFAR-10/100 (`classification` (python version))
  - [Format specification](https://www.cs.toronto.edu/~kriz/cifar.html)
  - [Dataset example CIFAR-10](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cifar10_dataset)
  - [Dataset example CIFAR-100](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cifar100_dataset)
  - [Format documentation](../explanation/formats/cifar)
- Cityscapes (`segmentation`)
  - [Format specification](https://www.cityscapes-dataset.com/dataset-overview/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cityscapes_dataset)
  - [Format documentation](../explanation/formats/cityscapes)
- Common Semantic Segmentation (`segmentation`)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/common_semantic_segmentation_dataset)
  - [Format documentation](../explanation/formats/common_semantic_segmentation)
- Common Super Resolution
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/common_super_resolution_dataset)
  - [Format documentation](../explanation/formats/common_super_resolution)
- CVAT (`for images`, `for video` (import-only))
  - [Format specification](https://opencv.github.io/cvat/docs/manual/advanced/xml_format)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/cvat_dataset)
- ICDAR13/15 (`word_recognition`, `text_localization`, `text_segmentation`)
  - [Format specification](https://rrc.cvc.uab.es/?ch=2)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/icdar_dataset)
- ImageNet (`classification`, `detection`)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/imagenet_dataset)
  - [Dataset example (txt for classification)](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/imagenet_txt_dataset)
  - Detection format is the same as in PASCAL VOC
  - [Format documentation](../explanation/formats/imagenet)
- KITTI (`segmentation`, `detection`)
  - [Format specification](http://www.cvlibs.net/datasets/kitti/index.php)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kitti_dataset)
  - [Format documentation](../explanation/formats/kitti)
- KITTI 3D (`raw`/`tracklets`/`velodyne points`)
  - [Format specification](http://www.cvlibs.net/datasets/kitti/raw_data.php)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kitti_dataset/kitti_raw)
  - [Format documentation](../explanation/formats/kitti_raw)
- Kinetics 400/600/700
  - [Format specification](https://www.deepmind.com/open-source/kinetics)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/kinetics_dataset)
  - [Format documentation](../explanation/formats/kinetics)
- LabelMe (`labels`, `boxes`, `masks`)
  - [Format specification](http://labelme.csail.mit.edu/Release3.0)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/labelme_dataset)
- LFW (`classification`, `person re-identification`, `landmarks`)
  - [Format specification](http://vis-www.cs.umass.edu/lfw/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/lfw_dataset)
  - [Format documentation](../explanation/formats/lfw)
- Mapillary Vistas (import-only)
  - [Format specification](https://www.mapillary.com/dataset/vistas)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mapillary_vistas_dataset)
  - [Format documentation](../explanation/formats/mapillary_vistas)
- Market-1501 (`person re-identification`)
  - [Format specification](https://www.aitribune.com/dataset/2018051063)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/market1501_dataset)
- MARS (import-only)
  - [Format specification](https://zheng-lab.cecs.anu.edu.au/Project/project_mars.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mars_dataset)
  - [Format documentation](../explanation/formats/mars)
- MNIST (`classification`)
  - [Format specification](http://yann.lecun.com/exdb/mnist/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mnist_dataset)
  - [Format documentation](../explanation/formats/mnist)
- MNIST in CSV (`classification`)
  - [Format specification](https://pjreddie.com/projects/mnist-in-csv/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mnist_csv_dataset)
  - [Format documentation](../explanation/formats/mnist)
- MOT sequences
  - [Format specification](https://arxiv.org/pdf/1906.04567.pdf)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mot_dataset)
- MOTS (png)
  - [Format specification](https://www.vision.rwth-aachen.de/page/mots)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mots_dataset)
- MPII Human Pose Dataset (`detection`, `pose estimation`) (import-only)
  - [Format specification](http://human-pose.mpi-inf.mpg.de)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mpii_dataset)
  - [Format documentation](../explanation/formats/mpii)
- MPII Human Pose Dataset (JSON) (`detection`, `pose estimation`) (import-only)
  - [Format specification](http://human-pose.mpi-inf.mpg.de)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/mpii_json_dataset)
  - [Format documentation](../explanation/formats/mpii_json)
- MS COCO (`image_info`, `instances`, `person_keypoints`, `captions`, `labels`,`panoptic`, `stuff`)
  - [Format specification](http://cocodataset.org/#format-data)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/coco_dataset)
  - `labels` are our extension - like `instances` with only `category_id`
  - [Format documentation](../explanation/formats/coco)
- NYU Depth Dataset V2 (`depth estimation`) (import-only)
  - [Format specification](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/nyu_depth_v2_dataset)
  - [Format documentation](../explanation/formats/nyu_depth_v2)
- Open Images (`classification`, `detection`, `segmentation`)
  - [Format specification](https://storage.googleapis.com/openimages/web/download.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/open_images_dataset)
  - [Format documentation](../explanation/formats/open_images)
- PASCAL VOC (`classification`, `detection`, `segmentation` (class, instances),
  `action_classification`, `person_layout`)
  - [Format specification](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/index.html)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/voc_dataset)
  - [Format documentation](../explanation/formats/pascal_voc)
- Supervisely (`pointcloud`)
  - [Format specification](https://docs.supervise.ly/data-organization/00_ann_format_navi)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/sly_pointcloud_dataset)
  - [Format documentation](../explanation/formats/sly_pointcloud)
- SYNTHIA (`segmentation`) (import-only)
  - [Format specification](https://synthia-dataset.net/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/synthia_dataset)
  - [Format documentation](../explanation/formats/synthia)
- TF Detection API (`bboxes`, `masks`)
  - Format specifications: [bboxes](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md),
    [masks](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/tf_detection_api_dataset)
- VGGFace2 (`landmarks`, `bboxes`)
  - [Format specification](https://github.com/ox-vgg/vgg_face2)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vgg_face2_dataset)
  - [Format documentation](../explanation/formats/vgg_face2)
- VoTT CSV (`detection`) (import-only)
  - [Format specification](https://github.com/microsoft/VoTT)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vott_csv_dataset)
  - [Format documentation](../explanation/formats/vott_csv)
- VoTT JSON (`detection`) (import-only)
  - [Format specification](https://github.com/microsoft/VoTT)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/vott_json_dataset)
  - [Format documentation](../explanation/formats/vott_json)
- WIDER Face (`bboxes`)
  - [Format specification](http://shuoyang1213.me/WIDERFACE/)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/widerface_dataset)
  - [Format documentation](../explanation/formats/wider_face)
- YOLO (`bboxes`)
  - [Format specification](https://github.com/AlexeyAB/darknet#how-to-train-pascal-voc-data)
  - [Dataset example](https://github.com/openvinotoolkit/datumaro/tree/develop/tests/assets/yolo_dataset)
  - [Format documentation](../explanation/formats/yolo)
- YOLO-Ultralytics (`bboxes`)
  - [Format specification](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml)
  - [Dataset example](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml)
  - [Format documentation](../explanation/formats/yolo_ultralytics)

## Supported annotation types

- Labels
- Bounding boxes
- Polygons
- Polylines
- (Segmentation) Masks
- (Key-)Points
- Captions
- 3D cuboids
- Super Resolution Annotation
- Depth Annotation
- Ellipses
- Hash Keys

Datumaro does not separate datasets by tasks like classification, detection
etc. Instead, datasets can have any annotations. When a dataset is exported
in a specific format, only relevant annotations are exported.

## Dataset meta info file

It is possible to use classes that are not original to the format.
To do this, use `dataset_meta.json`.

```json
{
    "label_map": {"0": "background", "1": "car", "2": "person"},
    "segmentation_colors": [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
    "background_label": "0"
}
```

- `label_map` is a dictionary where the class ID is the key and
  the class name is the value.
- `segmentation_colors` is a list of channel-wise values for each class.
  This is only necessary for the segmentation task.
- `background_label` is a background label ID in the dataset.
