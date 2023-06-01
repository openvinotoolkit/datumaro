# Copyright (C) 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.transformer import Transform
from datumaro.util import take_by


# pylint: disable=no-self-use
class Launcher(CliPlugin):
    def __init__(self, model_dir=None):
        pass

    def launch(self, inputs):
        raise NotImplementedError()

    def infos(self):
        return None

    def categories(self):
        return None

    def type_check(self, item):
        return True


# pylint: enable=no-self-use


class ModelTransform(Transform):
    def __init__(self, extractor, launcher, batch_size=1, append_annotation=False):
        super().__init__(extractor)
        self._launcher = launcher
        self._batch_size = batch_size
        self._append_annotation = append_annotation

    def __iter__(self):
        for batch in take_by(self._extractor, self._batch_size):
            inputs = []
            for item in batch:
                if not self._launcher.type_check(item):
                    continue
                inputs.append(np.atleast_3d(item.media.data))
            inputs = np.array(inputs)
            inference = self._launcher.launch(inputs)

            for item, annotations in zip(batch, inference):
                self._check_annotations(annotations)
                if self._append_annotation:
                    annotations = item.annotations + annotations
                yield self.wrap_item(item, annotations=annotations)

    def get_subset(self, name):
        subset = self._extractor.get_subset(name)
        return __class__(subset, self._launcher, self._batch_size)

    def infos(self):
        launcher_override = self._launcher.infos()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.infos()

    def categories(self):
        launcher_override = self._launcher.categories()
        if launcher_override is not None:
            return launcher_override
        return self._extractor.categories()

    def transform_item(self, item):
        inputs = np.expand_dims(item.media, axis=0)
        annotations = self._launcher.launch(inputs)[0]
        return self.wrap_item(item, annotations=annotations)

    def _check_annotations(self, annotations):
        labels_count = len(self.categories().get(AnnotationType.label, LabelCategories()).items)

        for ann in annotations:
            label = getattr(ann, "label", None)
            if label is None:
                continue

            if label not in range(labels_count):
                raise Exception(
                    "Annotation has unexpected label id %s, "
                    "while there is only %s defined labels." % (label, labels_count)
                )
