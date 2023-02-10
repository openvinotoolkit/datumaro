# Copyright (C) 2020-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import ConflictingCategoriesError, MediaTypeError


class Merger(CliPlugin):
    def __init__(self, **options):
        super().__init__(**options)
        self.__dict__["_sources"] = None

    def _merge_infos(self, sources):
        infos = {}
        for source in sources:
            for k, v in source.items():
                if k in infos:
                    log.warning(
                        "Duplicated infos field %s: overwrite from %s to %s", k, infos[k], v
                    )
                infos[k] = v
        return infos

    def _merge_categories(self, sources):
        categories = {}
        for source_idx, source in enumerate(sources):
            for cat_type, source_cat in source.items():
                existing_cat = categories.setdefault(cat_type, source_cat)
                if existing_cat != source_cat and len(source_cat) != 0:
                    if len(existing_cat) == 0:
                        categories[cat_type] = source_cat
                    else:
                        raise ConflictingCategoriesError(
                            "Merging of datasets with different categories is "
                            "only allowed in 'merge' command.",
                            sources=list(range(source_idx)),
                        )
        return categories

    def _merge_media_types(self, sources):
        if sources:
            media_type = sources[0].media_type()
            for s in sources:
                if not issubclass(s.media_type(), media_type) or not issubclass(
                    media_type, s.media_type()
                ):
                    # Symmetric comparision is needed in the case of subclasses:
                    # eg. Image and ByteImage
                    raise MediaTypeError("Datasets have different media types")
            return media_type

        return None


def get_merger(key):
    if key == "union":
        from datumaro.components.operations import UnionMerge

        merger = UnionMerge()
    elif key == "intersect":
        from datumaro.components.operations import IntersectMerge

        merger = IntersectMerge()
    else:
        from datumaro.components.operations import ExactMerge

        merger = ExactMerge()

    return merger
