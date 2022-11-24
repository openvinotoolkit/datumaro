# Copyright (C) 2022 Intel Corporationworkbench.action.openLargeOutput
#
# SPDX-License-Identifier: MIT



import os
import datumaro as dm
import time

from datumaro.components.searcher import Searcher

# dataset = dm.Dataset.import_from('./tests/assets/widerface_dataset')
start_time = time.time()
dataset = dm.Dataset.import_from('./tests/assets/brats_dataset', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/kitti_dataset/kitti_raw', format='kitti_raw', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/mapillary_vistas_dataset/dataset_with_meta_file', 'mapillary_vistas', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/mars_dataset', 'mars', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/voc_dataset/voc_dataset2', 'voc', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/common_semantic_segmentation_dataset/non_standard_dataset/segmentation', image_prefix="image_", mask_prefix="gt_", save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/cvat_dataset/for_video', "cvat", save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/datumaro_dataset', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/imagenet_txt_dataset/custom_labels', "imagenet_txt", labels_file="synsets-alt.txt", save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/kinetics_dataset', save_hash=True)
# dataset = dm.Dataset.import_from('./tests/assets/cifar10_dataset', format="cifar", save_hash=True)
# dataset = dm.Dataset.import_from('/media/hdd1/ade20k_val', format='common_semantic_segmentation', save_hash=True)
# dataset = dm.Dataset.import_from("coco_dataset", format='coco_instances', save_hash=True)
# dataset = dm.Dataset.import_from("//media/hdd1/Datasets/mfnd")
print(f'setting dataset time for {len(dataset)} items: ', time.time()-start_time)

for item in dataset:
    print(item)

for i, item in enumerate(dataset):
    if i == 1:
        query = item

# searcher = Searcher(dataset)
# topk_list = searcher.search_topk(query, topk=1)
# print(topk_list)
