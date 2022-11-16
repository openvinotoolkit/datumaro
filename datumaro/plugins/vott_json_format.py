# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp

from datumaro.components.annotation import AnnotationType, Bbox, LabelCategories
from datumaro.components.extractor import DatasetItem, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.components.media import Image
from datumaro.util import parse_json_file
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file


class VottJsonPath:
    ANNO_FILE_SUFFIX = "-export.json"


class VottJsonExtractor(SourceExtractor):
    def __init__(self, path):
        if not osp.isfile(path):
            raise FileNotFoundError("Can't read annotation file '%s'" % path)

        super().__init__(subset=osp.splitext(osp.basename(path))[0].rsplit("-", maxsplit=1)[0])

        if has_meta_file(path):
            self._categories = {
                AnnotationType.label: LabelCategories.from_iterable(parse_meta_file(path).keys())
            }
        else:
            self._categories = {AnnotationType.label: LabelCategories()}

        self._items = list(self._load_items(path).values())

    def _load_items(self, path):
        anno_dict = parse_json_file(path)

        label_categories = self._categories[AnnotationType.label]
        tags = anno_dict.get("tags", [])
        for label in tags:
            label_name = label.get("name")
            label_idx = label_categories.find(label_name)[0]
            if label_idx is None:
                label_idx = label_categories.add(label_name)

        items = {}
        for id, asset in anno_dict.get("assets", {}).items():
            item_id = osp.splitext(asset.get("asset", {}).get("name"))[0]
            annotations = []
            for region in asset.get("regions", []):
                tags = region.get("tags", [])
                if not tags:
                    bbox = region.get("boundingBox", {})
                    if bbox:
                        annotations.append(
                            Bbox(
                                float(bbox["left"]),
                                float(bbox["top"]),
                                float(bbox["width"]),
                                float(bbox["height"]),
                                attributes={"id": region.get("id")},
                            )
                        )

                for tag in region.get("tags", []):
                    label_idx = label_categories.find(tag)[0]
                    if label_idx is None:
                        label_idx = label_categories.add(tag)

                    bbox = region.get("boundingBox", {})
                    if bbox:
                        annotations.append(
                            Bbox(
                                float(bbox["left"]),
                                float(bbox["top"]),
                                float(bbox["width"]),
                                float(bbox["height"]),
                                label=label_idx,
                                attributes={"id": region.get("id")},
                            )
                        )

            items[item_id] = DatasetItem(
                id=item_id,
                subset=self._subset,
                attributes={"id": id},
                media=Image(path=osp.join(osp.dirname(path), asset.get("asset", {}).get("path"))),
                annotations=annotations,
            )

        return items


class VottJsonImporter(Importer):
    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".json", "vott_json")

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("*" + VottJsonPath.ANNO_FILE_SUFFIX)
