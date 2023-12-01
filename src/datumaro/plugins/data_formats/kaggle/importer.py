# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer


class KaggleRelaxedVocImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        with context.require_any():
            with context.alternative():
                cls._check_ann_file(
                    context.require_file("**/" + cls.ANN_DIR_NAME + "*" + cls.FORMAT_EXT), context
                )

    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(
            fpath, "Requirements for the annotation file of voc format"
        ) as fp:
            cls._check_ann_file_impl(fp)


class KaggleRelaxedYoloImporter(Importer):
    pass

    # FORMAT = "kaggle_relaxed_voc"

    # @classmethod
    # def detect(
    #     cls,
    #     context: FormatDetectionContext,
    # ) -> FormatDetectionConfidence:
    #     with context.require_any():
    #         for task in cls._TASKS.keys():
    #             with context.alternative():
    #                 context.require_file(f"annotations/{task.name}_*.json")
    #     return FormatDetectionConfidence.LOW


#     @classmethod
#     def find_sources(cls, path):
#         subset_paths = glob(osp.join(path, "**", "*_*.json"), recursive=True)

#         subsets = {}
#         for subset_path in subset_paths:
#             ann_type = detect_coco_task(osp.basename(subset_path))
#             if ann_type is None and len(cls._TASKS) == 1:
#                 ann_type = list(cls._TASKS)[0]

#             if ann_type not in cls._TASKS:
#                 log.warning(
#                     "File '%s' was skipped, could't match this file "
#                     "with any of these tasks: %s"
#                     % (subset_path, ",".join(e.NAME for e in cls._TASKS.values()))
#                 )
#                 continue

#             parts = osp.splitext(osp.basename(subset_path))[0].split(
#                 ann_type.name + "_", maxsplit=1
#             )
#             subset_name = parts[1] if len(parts) == 2 else DEFAULT_SUBSET_NAME
#             subsets.setdefault(subset_name, {})[ann_type] = subset_path

#         sources = []
#         for subset_path in subset_paths:
#             subset_name = osp.basename(osp.dirname(subset_path))
#             sources.append(
#                 {"url": subset_path, "format": cls.FORMAT, "options": {"subset": subset_name}}
#             )

#         return sources
