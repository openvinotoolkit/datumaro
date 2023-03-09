# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from copy import deepcopy
import csv
import google.protobuf.text_format as text_format
import os
import os.path as osp

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.errors import DatasetImportError
from datumaro.components.exporter import Exporter
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.os_util import find_files
import datumaro.plugins.data_formats.ava_label_pb2 as ava_label_pb2


class AvaPath:
    IMAGE_DIR = "frames"
    IMAGE_EXT = ".jpg"
    ANNOTATION_DIR = "annotations"
    ANNOTATION_EXT = ".txt"
    ANNOTATION_PREFIX = "ava_"
    ANNOTATION_VERSION = "_v2.2"
    CATEGORY_FILE = ANNOTATION_PREFIX + "action_list" + ANNOTATION_VERSION + ".pbtxt"


class AvaBase(SubsetBase):
    def __init__(self, path):
        if not osp.isfile(path):
            raise DatasetImportError(f"Can't find JSON file at '{path}'")
        self._path = path

        subset = osp.splitext(osp.basename(path))[0].replace(AvaPath.ANNOTATION_PREFIX, "").replace(AvaPath.ANNOTATION_VERSION, "")
        super().__init__(subset=subset)

        if path.endswith(osp.join(AvaPath.ANNOTATION_DIR, osp.basename(path))):
            self._rootpath = path.rsplit(AvaPath.ANNOTATION_DIR, maxsplit=1)[0]
        else:
            raise DatasetImportError(
                f"Annotation path ({path}) should be under the directory which is named {AvaPath.ANNOTATION_DIR}. "
                "If not, Datumaro fails to find the root path for this dataset."
            )

        if self._rootpath and osp.isdir(osp.join(self._rootpath, AvaPath.IMAGE_DIR)):
            self._images_dir = osp.join(self._rootpath, AvaPath.IMAGE_DIR)
        else:
            raise DatasetImportError(
                f"Root path ({self._rootpath}) should contain the directory which is named {AvaPath.IMAGE_DIR}. "
                "If not, Datumaro fails to find the image directory path."
            )

        category_path = osp.join(self._rootpath, AvaPath.ANNOTATION_DIR, AvaPath.CATEGORY_FILE)
        self._categories = self._load_categories(category_path)
        self._items = self._load_items(path)

    def _load_categories(self, category_path):
        with open(category_path, 'r') as f:
            pbtxt_data = f.read()

        label_list = ava_label_pb2.LabelList()
        text_format.Parse(pbtxt_data, label_list)

        categories = LabelCategories()
        categories.add('no action') # dummy class for id 0 for ava data
        for node in label_list.label:
            categories.add(node.name)

        return {AnnotationType.label: categories}
            
    def _load_items(self, ann_file):
        items = {}
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line_split = line.strip().split(",")

                video_id = line_split[0]
                timestamp = line_split[1].zfill(6)  # 6-digits

                item_id = video_id + "/" + timestamp
                image_path = osp.join(self._images_dir, video_id, item_id.replace("/", "_"), AvaPath.IMAGE_EXT)

                item = items.get(item_id)
                if item is None:
                    item = DatasetItem(
                        id=item_id,
                        subset=self._subset,
                        media=Image(path=image_path),
                    )
                    items[item_id] = item
                
                if "excluded_timestamps" in self._subset:
                    continue
                
                bbox = list(map(float, line_split[2:6]))
                label = int(line_split[6])
                entity_id = int(line_split[7])

                anns = item.annotations
                anns.append(
                    Bbox(
                        x=bbox[0],
                        y=bbox[2],
                        w=bbox[1],
                        h=bbox[3],
                        label=label,
                        attributes={"track_id": entity_id},
                    )
                )

        return items.values()


class AvaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        ann_files = find_files(path, exts="csv", recursive=True, max_depth=1)
        
        sources = []
        for ann_file in ann_files:
            sources.append({"url": ann_file, "format": AvaBase.NAME})
        
        return sources


class AvaExporter(Exporter):
    DEFAULT_IMAGE_EXT = AvaPath.IMAGE_EXT
    def apply(self):
        # if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
        #     raise MediaTypeError("Media type is not an image")

        save_dir = self._save_dir

        ann_dir = osp.join(save_dir, AvaPath.ANNOTATION_DIR)
        os.makedirs(ann_dir, exist_ok=True)
        
        frame_dir = osp.join(save_dir, AvaPath.IMAGE_DIR)

        # if self._save_dataset_meta:
        #     label_categories = self._extractor.categories()[AnnotationType.label]
        #     self._save_meta_file(save_dir)


        for subset_name, subset in self._extractor.subsets().items():
            ann_file = osp.join(ann_dir, AvaPath.ANNOTATION_PREFIX + subset_name + AvaPath.ANNOTATION_VERSION + AvaPath.ANNOTATION_EXT)
            with open(ann_file, mode='w') as csvfile:
                csvwriter = csv.writer(csvfile)
                for item in subset:
                    item_row = item.id.split("/")

                    if self._save_media:
                        video_dir = osp.join(frame_dir, item_row[0])
                        os.makedirs(video_dir, exist_ok=True)
                        self._save_image(item, path=osp.join(video_dir, item.id.replace("/", "_") + AvaPath.IMAGE_EXT))

                    bboxes = [a for a in item.annotations if a.type == AnnotationType.bbox]
                    if bboxes:
                        for bbox in bboxes:
                            csvwriter.writerow(item_row + [bbox.x, bbox.w, bbox.y, bbox.h, bbox.label, bbox.attributes.get('track_id', 0)])
