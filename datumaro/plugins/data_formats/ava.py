# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import Bbox
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util.os_util import find_files


class AvaBase(DatasetBase):
    def __init__(self, path):
        assert osp.isdir(path), path
        self._path = path

        super().__init__()

        self._items = {}
        for ann_file in find_files(dirpath=path, exts="csv", recursive=True, max_depth=1):
            print("26", ann_file)
            self._load_items_from_csv(ann_file)

        print(self._items)

    def _load_items_from_csv(self, ann_file):
        subset = osp.splitext(osp.basename(ann_file))[0]
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line_split = line.strip().split(",")

                video_id = line_split[0]
                timestamp = line_split[1]  # 6-digits?

                bbox = list(map(float, line_split[2:6]))
                label = int(line_split[6])
                entity_id = int(line_split[7])

                item_id = video_id + "/" + timestamp
                image_path = osp.join(self._path, "frames", video_id, timestamp, ".jpg")

                item = self._items.get(item_id)
                if item is None:
                    item = DatasetItem(
                        id=item_id,
                        subset=subset,
                        media=Image(path=image_path),
                    )
                    self._items[item_id] = item

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


class AvaImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        if find_files(path, exts="csv", recursive=True, max_depth=1):
            return [{"url": path, "format": AvaBase.NAME}]
