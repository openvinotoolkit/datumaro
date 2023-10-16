# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import os
from functools import partial

import numpy as np
import pytest

from datumaro.components.dataset_base import DatasetItem
from datumaro.components.environment import Environment
from datumaro.components.importer import DatasetImportError
from datumaro.components.media import FromFileMixin, Image
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.arrow import ArrowExporter, ArrowImporter
from datumaro.plugins.transforms import Sort

from ....requirements import Requirements, mark_requirement

from tests.utils.test_utils import check_save_and_load, compare_datasets, compare_datasets_strict


# class ArrowDatasetTest:
#     @mark_requirement(Requirements.DATUM_GENERAL_REQ)
#     @pytest.mark.parametrize(
#         ["fxt_dataset"],
#         [
#             pytest.param("fxt_arrow_dataset", id="test_arrow_format_with_keep_in_memory"),
#         ],
#     )
#     def test_arrow_dataset_getitem(self, helper_tc, fxt_dataset, request):
#         fxt_dataset = request.getfixturevalue(fxt_dataset)

#         helper_tc.assertTrue(
#             len(fxt_dataset.column_names) < len(fxt_dataset.flatten().column_names)
#         )

#         # column name
#         for column_name in fxt_dataset.column_names:
#             _dataset = fxt_dataset[column_name]
#             helper_tc.assertEqual(len(_dataset.column_names), 1)
#             helper_tc.assertEqual(column_name, _dataset.column_names[0])

#         with pytest.raises(KeyError):
#             fxt_dataset["invalid column name"]

#         length = len(fxt_dataset)

#         # positive integer
#         for i in range(length):
#             item = fxt_dataset[i]
#             helper_tc.assertTrue(isinstance(item, dict))
#             helper_tc.assertEqual(item["id"], str(i))

#         # negative integer
#         for i in range(length):
#             item = fxt_dataset[i - length]
#             helper_tc.assertTrue(isinstance(item, dict))
#             helper_tc.assertEqual(item["id"], str(i))

#         # positive slice
#         items = fxt_dataset[0:100]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual([i["id"] for i in items], [str(i) for i in range(0, 100)])

#         # positive range
#         items = fxt_dataset[range(0, 100)]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual([i["id"] for i in items], [str(i) for i in range(0, 100)])

#         # negative slice
#         items = fxt_dataset[-100:-1]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual(
#             [i["id"] for i in items], [str(i) for i in range(length - 100, length - 1)]
#         )

#         # negative range
#         items = fxt_dataset[range(length - 100, length - 1)]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual(
#             [i["id"] for i in items], [str(i) for i in range(length - 100, length - 1)]
#         )

#         # positive slice with interval
#         items = fxt_dataset[0:100:3]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual([i["id"] for i in items], [str(i) for i in range(0, 100, 3)])

#         # positive range with interval
#         items = fxt_dataset[range(0, 100, 3)]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual([i["id"] for i in items], [str(i) for i in range(0, 100, 3)])

#         # negative slice with interval
#         items = fxt_dataset[-100:-1:3]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual(
#             [i["id"] for i in items], [str(i) for i in range(length - 100, length - 1, 3)]
#         )

#         # negative range with interval
#         items = fxt_dataset[range(length - 100, length - 1, 3)]
#         helper_tc.assertTrue(isinstance(items, list))
#         helper_tc.assertEqual(
#             [i["id"] for i in items], [str(i) for i in range(length - 100, length - 1, 3)]
#         )

#         with pytest.raises(KeyError):
#             fxt_dataset[0.1]

#         with pytest.raises(IndexError):
#             fxt_dataset[length]


class ArrowFormatTest:
    exporter = ArrowExporter
    importer = ArrowImporter
    format = ArrowImporter.NAME

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
            move_save_dir=False,
            **kwargs,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        [
            "fxt_dataset",
            "compare",
            "save_media",
            "require_media",
            "fxt_export_kwargs",
            "post_processing",
        ],
        [
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_and_load",
            ),
            pytest.param(
                "fxt_test_datumaro_format_dataset",
                None,
                False,
                False,
                {},
                None,
                id="test_can_save_and_load_with_no_save_media",
            ),
            pytest.param(
                "fxt_relative_paths",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_relative_paths",
            ),
            pytest.param(
                "fxt_can_save_dataset_with_cjk_categories",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_dataset_with_cjk_categories",
            ),
            pytest.param(
                "fxt_can_save_dataset_with_cyrillic_and_spaces_in_filename",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_dataset_with_cyrillic_and_spaces_in_filename",
            ),
            pytest.param(
                "fxt_can_save_and_load_image_with_arbitrary_extension",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_and_load_image_with_arbitrary_extension",
            ),
            pytest.param(
                "fxt_can_save_and_load_infos",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_and_load_infos",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_and_load_image",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets_strict,
                True,
                True,
                {"image_ext": "PNG"},
                None,
                id="test_can_save_and_load_image_with_png_scheme",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets_strict,
                True,
                True,
                {"image_ext": "TIFF"},
                None,
                id="test_can_save_and_load_image_with_tiff_scheme",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets,
                True,
                False,
                {"image_ext": "JPEG/75"},
                None,
                id="test_can_save_and_load_image_with_jpeg_75_scheme",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets,
                True,
                False,
                {"image_ext": "JPEG/95"},
                None,
                id="test_can_save_and_load_image_with_jpeg_95_scheme",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets_strict,
                True,
                True,
                {"max_chunk_size": None, "num_shards": 5},
                lambda dataset: Sort(dataset, lambda item: int(item.id)),
                id="test_can_save_and_load_image_with_num_shards",
            ),
            pytest.param(
                "fxt_image",
                compare_datasets_strict,
                True,
                True,
                {"max_chunk_size": 20, "max_shard_size": "1M"},
                lambda dataset: Sort(dataset, lambda item: int(item.id)),
                id="test_can_save_and_load_image_with_max_size",
            ),
            pytest.param(
                "fxt_point_cloud",
                compare_datasets_strict,
                True,
                True,
                {},
                None,
                id="test_can_save_and_load_point_cloud",
            ),
            pytest.param(
                "fxt_point_cloud",
                compare_datasets_strict,
                True,
                True,
                {"max_chunk_size": 20, "num_shards": 5},
                lambda dataset: Sort(dataset, lambda item: int(item.id)),
                id="test_can_save_and_load_point_cloud_with_num_shards",
            ),
            pytest.param(
                "fxt_point_cloud",
                compare_datasets_strict,
                True,
                True,
                {"max_chunk_size": 20, "max_shard_size": "1M"},
                lambda dataset: Sort(dataset, lambda item: int(item.id)),
                id="test_can_save_and_load_point_cloud_with_max_size",
            ),
            pytest.param(
                "fxt_large",
                compare_datasets_strict,
                True,
                True,
                {"max_chunk_size": 1000},
                lambda dataset: Sort(dataset, lambda item: int(item.id)),
                id="test_large",
            ),
        ],
    )
    def test_can_save_and_load(
        self,
        fxt_dataset,
        compare,
        save_media,
        require_media,
        test_dir,
        fxt_import_kwargs,
        fxt_export_kwargs,
        post_processing,
        helper_tc,
        request,
    ):
        fxt_dataset = request.getfixturevalue(fxt_dataset)
        self._test_save_and_load(
            helper_tc,
            fxt_dataset,
            partial(self.exporter.convert, save_media=save_media, **fxt_export_kwargs),
            test_dir,
            compare=compare,
            require_media=require_media,
            importer_args=fxt_import_kwargs,
            post_processing=post_processing,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self, fxt_test_datumaro_format_dataset, test_dir):
        self.exporter.convert(fxt_test_datumaro_format_dataset, save_dir=test_dir)

        detected_formats = Environment().detect_dataset(test_dir)
        assert [self.importer.NAME] == detected_formats

    @pytest.mark.parametrize(
        ["fxt_dataset", "save_media"],
        [
            pytest.param(
                "fxt_image",
                True,
                id="image_with_media",
            ),
            pytest.param(
                "fxt_point_cloud",
                True,
                id="point_cloud_with_media",
            ),
            pytest.param(
                "fxt_image",
                False,
                id="image_without_media",
            ),
            pytest.param(
                "fxt_point_cloud",
                False,
                id="point_cloud_without_media",
            ),
        ],
    )
    def test_media_contents(self, fxt_dataset, save_media, test_dir, request):
        fxt_dataset = request.getfixturevalue(fxt_dataset)

        fxt_dataset.export(test_dir, format=self.format, save_media=save_media)
        imported_dataset = Dataset.import_from(test_dir)
        for item_a, item_b in zip(fxt_dataset, imported_dataset):
            if isinstance(item_a.media, FromFileMixin):
                assert item_a.media.bytes is not None
            assert item_a.media.data is not None
            if save_media:
                assert item_b.media.bytes is not None
                assert item_b.media.data is not None
            else:
                if isinstance(item_a.media, FromFileMixin):
                    assert item_b.media.bytes is not None
                    assert item_b.media.data is not None
                else:
                    assert item_b.media.bytes is None
                    assert item_b.media.data is None

    # Below is testing special cases...
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_direct_changes(self, test_dir, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))),
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
                DatasetItem(3, subset="c", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
            ]
        )
        dataset.export(test_dir, format=self.format, save_media=True)

        dataset.put(DatasetItem(2, subset="a", media=Image.from_numpy(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "c")
        dataset.save(save_media=True)

        compare_datasets_strict(
            helper_tc, expected, Dataset.import_from(test_dir, format=self.format)
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_transforms(self, test_dir, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(2, subset="test"),
                DatasetItem(3, subset="train", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
                DatasetItem(4, subset="test", media=Image.from_numpy(data=np.ones((2, 3, 3)))),
            ],
            media_type=Image,
        )
        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="b"),
                DatasetItem(3, subset="c", media=Image.from_numpy(data=np.ones((2, 2, 3)))),
                DatasetItem(4, subset="d", media=Image.from_numpy(data=np.ones((2, 3, 3)))),
            ],
            media_type=Image,
        )

        dataset.export(test_dir, format=self.format, save_media=True)

        dataset.filter("/item[id >= 2]")
        dataset.transform("random_split", splits=(("train", 0.5), ("test", 0.5)), seed=42)
        dataset.save(save_media=True)

        compare_datasets_strict(
            helper_tc, expected, Dataset.import_from(test_dir, format=self.format)
        )
