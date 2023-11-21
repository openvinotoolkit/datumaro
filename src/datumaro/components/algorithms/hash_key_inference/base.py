# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, Sequence

from datumaro.components.dataset import Dataset

if TYPE_CHECKING:
    import datumaro.plugins.explorer as explorer
else:
    from datumaro.util.import_util import lazy_import

    explorer = lazy_import("datumaro.plugins.explorer")


class HashInference:
    def __init__(self, *datasets: Sequence[Dataset]) -> None:
        pass

    @property
    def model(self):
        if self._model is None:
            self._model = explorer.ExplorerLauncher(model_name="clip_visual_ViT-B_32")
        return self._model

    @property
    def text_model(self):
        if self._text_model is None:
            self._text_model = explorer.ExplorerLauncher(model_name="clip_text_ViT-B_32")
        return self._text_model

    def _compute_hash_key(self, datasets, datasets_to_infer):
        for dataset_to_infer in datasets_to_infer:
            if dataset_to_infer:
                dataset_to_infer.run_model(self.model, append_annotation=True)
        for dataset, dataset_to_infer in zip(datasets, datasets_to_infer):
            updated_items = [
                dataset.get(item.id, item.subset).wrap(annotations=item.annotations)
                for item in dataset_to_infer
            ]
            dataset.update(updated_items)
        return datasets
