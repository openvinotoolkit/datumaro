# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
from functools import partial

import numpy as np
import pytest

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.importer import DatasetImportError
from datumaro.components.media import Image
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.datumaro.exporter import DatumaroExporter
from datumaro.plugins.data_formats.datumaro.format import DatumaroPath
from datumaro.plugins.data_formats.datumaro.importer import DatumaroImporter

from ....requirements import Requirements, mark_requirement

from tests.utils.test_utils import (
    Dimensions,
    check_save_and_load,
    compare_datasets,
    compare_datasets_strict,
)


class DatumaroFormatTest:
    exporter = DatumaroExporter
    importer = DatumaroImporter
    format = DatumaroImporter.NAME
    ann_ext = DatumaroPath.ANNOTATION_EXT

    def _test_save_and_load(
        self,
        helper_tc,
        source_dataset,
        converter,
        test_dir,
        target_dataset=None,
        importer_args=None,
        compare=compare_datasets_strict,
        **kwargs,
    ):
        return check_save_and_load(
            helper_tc,
            source_dataset,
            converter,
            test_dir,
            importer=self.importer.NAME,
            target_dataset=target_dataset,
            importer_args=importer_args,
            compare=compare,
            move_save_dir=True,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt_dataset, compare, require_media",
        [
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                id="test_can_save_and_load",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                None,
                False,
                id="test_can_save_and_load_with_no_save_media",
            ),
            pytest.param(
                "fxt_relative_paths",
                compare_datasets_strict,
                True,
                id="test_relative_paths",
            ),
            pytest.param(
                "fxt_can_save_dataset_with_cjk_categories",
                compare_datasets_strict,
                True,
                id="test_can_save_dataset_with_cjk_categories",
            ),
            pytest.param(
                "fxt_can_save_dataset_with_cyrillic_and_spaces_in_filename",
                compare_datasets_strict,
                True,
                id="test_can_save_dataset_with_cyrillic_and_spaces_in_filename",
            ),
            pytest.param(
                "fxt_can_save_and_load_image_with_arbitrary_extension",
                compare_datasets_strict,
                True,
                id="test_can_save_and_load_image_with_arbitrary_extension",
            ),
            pytest.param(
                "fxt_can_save_and_load_infos",
                compare_datasets_strict,
                True,
                id="test_can_save_and_load_infos",
            ),
        ],
    )
    def test_can_save_and_load(
        self,
        fxt_dataset,
        compare,
        require_media,
        test_dir,
        fxt_import_kwargs,
        fxt_export_kwargs,
        helper_tc,
        request,
    ):
        fxt_dataset = request.getfixturevalue(fxt_dataset)
        self._test_save_and_load(
            helper_tc,
            fxt_dataset,
            partial(self.exporter.convert, save_media=True, **fxt_export_kwargs),
            test_dir,
            compare=compare,
            require_media=require_media,
            importer_args=fxt_import_kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "fxt_dataset_pair, compare, require_media, dimension",
        [
            pytest.param(
                "fxt_point_cloud_dataset_pair",
                None,
                True,
                Dimensions.dim_3d,
                id="test_can_save_and_load_point_cloud",
            ),
            pytest.param(
                "fxt_legacy_dataset_pair",
                None,
                True,
                Dimensions.dim_2d,
                id="test_can_save_and_load_legacy",
            ),
        ],
    )
    def test_source_target_pair(
        self, fxt_dataset_pair, compare, require_media, dimension, test_dir, helper_tc, request
    ):
        source_dataset, target_dataset = request.getfixturevalue(fxt_dataset_pair)
        self._test_save_and_load(
            helper_tc,
            source_dataset,
            partial(self.exporter.convert, save_media=True),
            test_dir,
            target_dataset,
            compare=compare,
            require_media=require_media,
            dimension=dimension,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self, fxt_test_datumaro_format_dataset, test_dir):
        self.exporter.convert(fxt_test_datumaro_format_dataset, save_dir=test_dir)

        detected_formats = Environment().detect_dataset(test_dir)
        assert [self.importer.NAME] == detected_formats

    # Below is testing special cases...
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_direct_changes(self, test_dir, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b"),
            ]
        )

        # generate initial dataset
        dataset = Dataset.from_iterable(
            [
                # modified subset
                DatasetItem(1, subset="a"),
                # unmodified subset
                DatasetItem(2, subset="b"),
                # removed subset
                DatasetItem(3, subset="c", media=Image(data=np.ones((2, 2, 3)))),
            ]
        )
        dataset.export(test_dir, format=self.format, save_media=True)

        dataset.put(DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "c")
        dataset.save(save_media=True)

        helper_tc.assertEqual(
            {f"a{self.ann_ext}", f"b{self.ann_ext}"},
            set(os.listdir(osp.join(test_dir, "annotations"))),
        )
        helper_tc.assertEqual({"2.jpg"}, set(os.listdir(osp.join(test_dir, "images", "a"))))
        compare_datasets_strict(
            helper_tc, expected, Dataset.import_from(test_dir, format=self.format)
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_transforms(self, test_dir, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(2, subset="test"),
                DatasetItem(3, subset="train", media=Image(data=np.ones((2, 2, 3)))),
                DatasetItem(4, subset="test", media=Image(data=np.ones((2, 3, 3)))),
            ],
            media_type=Image,
        )
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="b"),
                DatasetItem(3, subset="c", media=Image(data=np.ones((2, 2, 3)))),
                DatasetItem(4, subset="d", media=Image(data=np.ones((2, 3, 3)))),
            ],
            media_type=Image,
        )

        dataset.export(test_dir, format=self.format, save_media=True)

        dataset.filter("/item[id >= 2]")
        dataset.transform("random_split", splits=(("train", 0.5), ("test", 0.5)), seed=42)
        dataset.save(save_media=True)

        helper_tc.assertEqual({"images", "annotations"}, set(os.listdir(test_dir)))
        helper_tc.assertEqual(
            {f"train{self.ann_ext}", f"test{self.ann_ext}"},
            set(os.listdir(osp.join(test_dir, "annotations"))),
        )
        helper_tc.assertEqual({"3.jpg"}, set(os.listdir(osp.join(test_dir, "images", "train"))))
        helper_tc.assertEqual({"4.jpg"}, set(os.listdir(osp.join(test_dir, "images", "test"))))
        helper_tc.assertEqual(
            {"train", "c", "d", "test"}, set(os.listdir(osp.join(test_dir, "images")))
        )
        helper_tc.assertEqual(set(), set(os.listdir(osp.join(test_dir, "images", "c"))))
        helper_tc.assertEqual(set(), set(os.listdir(osp.join(test_dir, "images", "d"))))

        compare_datasets(helper_tc, expected, Dataset.import_from(test_dir, format=self.format))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_version_compatibility(self, fxt_wrong_version_dir):
        with pytest.raises(DatasetImportError):
            Dataset.import_from(fxt_wrong_version_dir, "datumaro")
