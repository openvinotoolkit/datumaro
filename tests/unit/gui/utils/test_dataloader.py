# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import tempfile
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from datumaro import Dataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.plugins.validators import ClassificationValidator

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path

cwd = os.getcwd()

import sys

sys.path.append(os.path.join(cwd, "gui"))

from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

from gui.datumaro_gui.utils.dataset.data_loader import DataRepo, DatasetHelper


class DataRepoTest:
    def setup_method(self):
        self.data_repo = DataRepo()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_dataset_dir(self):
        file_id = "test_file_id"
        directory = self.data_repo.get_dataset_dir(file_id)

        assert os.path.exists(directory)
        assert directory.endswith(file_id + os.sep + self.data_repo._dataset_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_file(self):
        file_content = b"test content"
        uploaded_file = UploadedFile(
            record=UploadedFileRec(
                file_id="test_file_id", name="test_file.txt", type="text/plain", data=file_content
            ),
            file_urls="http://example.com",
        )

        file_path = self.data_repo.save_file(uploaded_file)
        assert os.path.exists(file_path)

        with open(file_path, "rb") as f:
            assert f.read() == file_content


class DatasetHelperTest:
    @pytest.fixture
    def dataset_helper(self):
        dataset_dir = get_test_asset_path("datumaro_dataset")
        helper = DatasetHelper(dataset_root=dataset_dir)
        yield helper

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_init(self, dataset_helper):
        assert dataset_helper._dataset_dir
        assert not dataset_helper._detected_formats
        assert not dataset_helper._dm_dataset
        assert dataset_helper._format == ""
        assert dataset_helper._val_reports == {}
        assert not dataset_helper._image_stats
        assert not dataset_helper._ann_stats
        assert not dataset_helper._image_size_info
        assert dataset_helper._xml_items == {}
        assert dataset_helper._subset_to_ids == {}

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_init_dependent_variables(self, dataset_helper):
        # Ensure dependent variables are initialized
        dataset_helper._image_stats = {}
        dataset_helper._init_dependent_variables()
        assert not dataset_helper._image_stats

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @patch.object(DEFAULT_ENVIRONMENT, "detect_dataset", return_value=["datumaro"])
    def test_detect_format(self, mock_detect_dataset, dataset_helper):
        result = dataset_helper.detect_format()
        assert result == ["datumaro"]
        mock_detect_dataset.assert_called_once_with(path=dataset_helper._dataset_dir)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @patch.object(Dataset, "import_from")
    def test_import_dataset(self, mock_import_from, dataset_helper):
        format_ = "datumaro"
        result = dataset_helper.import_dataset(format_)
        assert dataset_helper._format == format_
        assert dataset_helper._dm_dataset == mock_import_from.return_value

        mock_import_from.assert_called_once_with(path=dataset_helper._dataset_dir, format=format_)
        assert result == mock_import_from.return_value

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset(self, dataset_helper):
        # Ensure correct dataset is returned
        dataset = MagicMock()
        dataset_helper._dm_dataset = dataset
        assert dataset_helper.dataset() == dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_format(self, dataset_helper):
        # Ensure correct format is returned
        dataset_helper._format = "datumaro"
        assert dataset_helper.format() == "datumaro"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_subset_to_ids(self, dataset_helper):
        expected_result = {"test": ["c"], "train": ["a", "b"], "validation": ["d"]}
        # Set dm_dataset
        dataset_helper.import_dataset("datumaro")
        result = dataset_helper.subset_to_ids()
        assert result == expected_result

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @patch.object(ClassificationValidator, "validate", return_value={"accuracy": 0.95})
    def test_validate(self, mock_validate, dataset_helper):
        task = "classification"
        expected_report = {"accuracy": 0.95}
        result = dataset_helper.validate(task)

        assert result == expected_report
        mock_validate.assert_called_once_with(dataset_helper._dm_dataset)


class DataRepoTest:
    @pytest.fixture
    def data_repo(self):
        return DataRepo()

    @pytest.fixture
    def mock_fake_file(self):
        class FakeFile:
            def __init__(self, name):
                self.name = name

            def getbuffer(self):
                return b"Test file content"

        with patch(UploadedFile, return_value=FakeFile("test_file.txt")):
            yield

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_get_dataset_dir(self, data_repo):
        file_id = "example_file_id"
        result = data_repo.get_dataset_dir(file_id)
        expected_path = os.path.join(".data_repo", file_id, "dataset")
        assert result == expected_path
        assert os.path.exists(result)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_file(self, data_repo):
        uploaded_file_rec = UploadedFileRec(
            file_id="test_file_id", name="test_file.txt", type="file_type", data=np.ones((2, 3))
        )

        uploaded_file = UploadedFile(record=uploaded_file_rec, file_urls=FileURLsProto())

        result_path = data_repo.save_file(uploaded_file)
        assert os.path.exists(result_path)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_delete_by_id(self, data_repo):
        file_id = "test_file_id"
        directory = os.path.join(data_repo._root_path, file_id)
        if not os.path.exists(directory):
            os.makedirs(directory)
        assert os.path.exists(directory)

        data_repo.delete_by_id(file_id)
        assert not os.path.exists(directory)
