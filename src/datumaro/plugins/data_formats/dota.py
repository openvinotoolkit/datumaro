# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging as log
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Type, TypeVar

from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories, RotatedBbox
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetItem, IDataset, SubsetBase
from datumaro.components.dataset_item_storage import ItemStatus
from datumaro.components.errors import (
    DatasetExportError,
    DatasetImportError,
    InvalidAnnotationError,
    MediaTypeError,
)
from datumaro.components.exporter import Exporter
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import ImportContext, Importer
from datumaro.components.media import Image
from datumaro.components.task import TaskAnnotationMapping
from datumaro.util import str_to_bool
from datumaro.util.image import IMAGE_EXTENSIONS
from datumaro.util.os_util import find_files

T = TypeVar("T")


class DotaFormat:
    ANNOTATION_DIR = "labelTxt"
    IMAGE_DIR = "images"


class DotaBase(SubsetBase):
    def __init__(
        self,
        path: Optional[List[str]] = None,
        *,
        img_path: Optional[str] = None,
        subset: Optional[str] = None,
        ctx: Optional[ImportContext] = None,
    ) -> None:
        super().__init__(subset=subset, ctx=ctx)

        if not osp.isdir(path):
            raise DatasetImportError(f"Can't find annotation directory {path}")

        self._path = path

        super().__init__(subset=subset, ctx=ctx)

        self._img_files = self._load_img_files(img_path)
        self._label_categories = self._load_categories(path)
        self._categories = {AnnotationType.label: self._label_categories}

        self._items = self._load_items(path)
        self._task_type = TaskAnnotationMapping().get_task(self._ann_types)

    def _load_img_files(self, rootpath: str) -> Dict[str, str]:
        return {
            self._get_fname(img_file): img_file
            for img_file in find_files(rootpath, IMAGE_EXTENSIONS, recursive=True, max_depth=2)
        }

    def _load_categories(self, path):
        label_names = []
        for ann_file in os.listdir(path):
            label_names.extend(
                self._parse_annotations(
                    ann_file=osp.join(self._path, ann_file), only_label_names=True
                )
            )

        label_categories = LabelCategories()
        for label_name in sorted(set(label_names)):
            label_categories.add(label_name)

        return label_categories

    def _load_items(self, path):
        items = []
        for ann_file in os.listdir(path):
            fname = osp.splitext(ann_file)[0]
            img = Image.from_file(path=self._img_files[fname])
            anns = self._parse_annotations(
                ann_file=osp.join(self._path, ann_file), only_label_names=False
            )
            items.append(DatasetItem(id=fname, subset=self._subset, media=img, annotations=anns))
        return items

    def _get_fname(self, fpath: str) -> str:
        return osp.splitext(osp.basename(fpath))[0]

    def _parse_annotations(
        self,
        ann_file: str,
        only_label_names: bool,
    ) -> List[Annotation]:
        lines = []
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []
        for line in lines:
            parts = line.split()
            if len(parts) != 10:
                log.debug(
                    f"Unexpected field count {len(parts)} in the bbox description. "
                    "Expected 10 fields (8 coordinates for rectangle, category, and difficulty)."
                )
                continue

            label_name = self._parse_field(parts[-2], str, "label_name")

            if only_label_names:
                annotations.append(label_name)
                continue

            label_id, _ = self._label_categories.find(label_name)
            coords = [
                (
                    self._parse_field(parts[i], float, "coords"),
                    self._parse_field(parts[i + 1], float, "coords"),
                )
                for i in range(0, 8, 2)
            ]
            difficulty = self._parse_field(parts[-1], int, "difficulty")

            annotations.append(
                RotatedBbox.from_rectangle(
                    coords, label=label_id, attributes={"difficulty": difficulty}
                )
            )
            self._ann_types.add(AnnotationType.rotated_bbox)

        return annotations

    @staticmethod
    def _parse_field(value: str, desired_type: Type[T], field_name: str) -> T:
        try:
            return desired_type(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {desired_type}"
            ) from e


class DotaImporter(Importer):
    _ANNO_EXT = ".txt"

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        context.require_file("**/" + DotaFormat.ANNOTATION_DIR + "/*" + cls._ANNO_EXT)
        return FormatDetectionConfidence.MEDIUM

    # @classmethod
    # def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
    #     with context.require_any():
    #         with context.alternative():
    #             cls._check_ann_file(
    #                 context.require_file("**/" + cls.ANN_DIR_NAME + "*" + cls.FORMAT_EXT), context
    #             )

    #     return FormatDetectionConfidence.MEDIUM

    # @classmethod
    # def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
    #     with context.probe_text_file(
    #         fpath, "Requirements for the annotation file of voc format"
    #     ) as fp:
    #         cls._check_ann_file_impl(fp)

    # @classmethod
    # def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
    #     for line in fp:
    #         fields = line.rstrip("\n").split(" ")
    #         if len(fields) != 10:
    #             raise DatasetImportError(
    #                 f"Roboflow Yolo OBB format txt file should have 10 fields for each line, "
    #                 f"but the read line has {len(fields)} fields: fields={fields}."
    #             )

    #         # Check the first line only
    #         return True

    #     raise DatasetImportError("Empty file is not allowed.")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(
            path=path,
            ext=cls._ANNO_EXT,
            dirname=DotaFormat.ANNOTATION_DIR,
            extractor_name="dota",
        )

        data_paths = set()
        for source in sources:
            url = osp.dirname(source["url"])
            subset_name = osp.relpath(source["url"], path).split(osp.sep)[0]
            data_paths.add((subset_name, url))

        return [
            {
                "url": ann_dir,
                "format": "dota",
                "options": {
                    "subset": subset,
                    "img_path": osp.join(path, subset, DotaFormat.IMAGE_DIR),
                },
            }
            for subset, ann_dir in data_paths
        ]

    @classmethod
    def get_file_extensions(cls) -> List[str]:
        return [cls._ANNO_EXT]


class DotaExporter(Exporter):
    DEFAULT_IMAGE_EXT = ".png"

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--add-path-prefix",
            default=True,
            type=str_to_bool,
            help="Add the 'data/' prefix for paths in the dataset info (default: %(default)s)",
        )
        return parser

    def __init__(
        self, extractor: IDataset, save_dir: str, *, add_path_prefix: bool = True, **kwargs
    ) -> None:
        super().__init__(extractor, save_dir, **kwargs)

    def _apply_impl(self):
        extractor = self._extractor
        save_dir = self._save_dir

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(save_dir, exist_ok=True)

        label_categories = extractor.categories()[AnnotationType.label]

        subsets = self._extractor.subsets()
        for subset_name, subset in subsets.items():
            if not subset_name or subset_name == DEFAULT_SUBSET_NAME:
                subset_name = DEFAULT_SUBSET_NAME

            subset_dir = osp.join(save_dir, subset_name)
            os.makedirs(subset_dir, exist_ok=True)

            for item in subset:
                try:
                    self._export_media(item, subset_dir)
                    self._export_item_annotation(item, subset_dir, label_categories)

                except Exception as e:
                    self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    def _export_media(self, item: DatasetItem, subset_dir: str) -> str:
        try:
            if not item.media or not (item.media.has_data or item.media.has_size):
                raise DatasetExportError(
                    "Failed to export item '%s': " "item has no image info" % item.id
                )

            image_name = self._make_image_filename(item)
            image_fpath = osp.join(subset_dir, DotaFormat.IMAGE_DIR, image_name)

            if self._save_media:
                self._save_image(item, image_fpath)

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    def _export_item_annotation(
        self, item: DatasetItem, subset_dir: str, label_categories: LabelCategories
    ) -> None:
        try:
            annotations = ""
            for bbox in item.annotations:
                if not isinstance(bbox, RotatedBbox) or bbox.label is None:
                    continue
                coords = bbox.as_polygon()
                coords = " ".join("%.2f %.2f" % (x, y) for x, y in coords)
                label_name = label_categories[bbox.label].name
                difficulty = bbox.attributes.get("difficulty", 0)
                annotations += "%s %s %s\n" % (coords, label_name, difficulty)

            annotation_path = osp.join(subset_dir, DotaFormat.ANNOTATION_DIR, "%s.txt" % item.id)
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)

            with open(annotation_path, "w", encoding="utf-8") as f:
                f.write(annotations)

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
        conv = cls(dataset, save_dir=save_dir, **kwargs)
        conv._patch = patch
        conv.apply()

        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            if subset == DEFAULT_SUBSET_NAME:
                subset = DEFAULT_SUBSET_NAME
            subset_dir = osp.join(save_dir, "obj_%s_data" % subset)

            image_path = osp.join(subset_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.remove(image_path)

            ann_path = osp.join(subset_dir, "%s.txt" % item.id)
            if osp.isfile(ann_path):
                os.remove(ann_path)
