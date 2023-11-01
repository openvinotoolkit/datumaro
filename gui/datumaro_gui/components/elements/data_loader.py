# Copyright (C) 2019-2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import shutil
import zipfile

from streamlit.runtime.uploaded_file_manager import UploadedFile

from datumaro.components.dataset import Dataset
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.hl_ops import HLOps
from datumaro.plugins.validators import (
    ClassificationValidator,
    DetectionValidator,
    SegmentationValidator,
)


class DataRepo:
    """
    Implements the data repo for Single Dataset Projects
    """

    def __init__(self):
        self._root_path = ".data_repo"
        self._dataset_dir = "dataset"

    def get_dataset_dir(_self, file_id: str) -> str:
        """
        Get directory of unzipped dataset.

        :param file_id: fild_id of uploaded zip file
        :return: path to dataset directory
        """
        directory = os.path.join(_self._root_path, file_id, _self._dataset_dir)
        os.makedirs(directory, exist_ok=True)
        return directory

    def unzip_dataset(_self, uploaded_zip: UploadedFile) -> str:
        """
        Unzip uploaded zip file to a dataset directory

        :param uploaded_zip: uploaded zip file from streamlit ui
        :return: path to dataset directory
        """

        def find_dataset_root(filelist: list[str]):
            common_path = os.path.commonpath(filelist)
            while common_path + os.sep in filelist:
                filelist.remove(common_path + os.sep)
                common_path = os.path.commonpath(filelist)
            return common_path

        assert zipfile.is_zipfile(
            uploaded_zip
        )  # .type in ["application/zip", "application/x-zip-compressed"]

        with zipfile.ZipFile(uploaded_zip, "r") as z:
            directory = _self.get_dataset_dir(uploaded_zip.file_id)

            dataset_root = find_dataset_root(z.namelist())
            if dataset_root == "":
                z.extractall(directory)
            else:
                dataset_root = dataset_root + os.sep
                start = len(dataset_root)
                zipinfos = z.infolist()
                for zipinfo in zipinfos:
                    if len(zipinfo.filename) > start:
                        zipinfo.filename = zipinfo.filename[start:]
                        z.extract(zipinfo, directory)

        return directory

    def delete_by_id(_self, fild_id: str):
        """
        Delete (unzipped) dataset for a given file_id

        :param file_id: fild_id of uploaded zip file
        :raises: FileNotFoundError
        """
        path = os.path.join(_self._root_path, str(fild_id))
        if os.path.exists(path):
            print(f"delete {path}")
            shutil.rmtree(path)


class DatasetHelper:
    """
    Import dm_dataset from DataRepo
    """

    def __init__(self, dataset_root: str):
        self._dataset_dir = dataset_root
        self._detected_formats = None
        self._dm_dataset = None
        self._format = None
        self._val_reports = {}

    def __del__(self):
        file_id = os.path.basename(os.path.dirname(self._dataset_dir))
        DataRepo().delete_by_id(file_id)

    def detect_format(_self) -> list[str]:
        if _self._detected_formats is None:
            _self._detected_formats = DEFAULT_ENVIRONMENT.detect_dataset(path=_self._dataset_dir)
            print("formats from", _self._dataset_dir, ":", _self._detected_formats)
        return _self._detected_formats

    def import_dataset(_self, format) -> Dataset:
        if format != _self._format:
            _self._format = format
            _self._dm_dataset = Dataset.import_from(path=_self._dataset_dir, format=_self._format)
            _self._val_reports = {}
        return _self._dm_dataset

    def dataset(_self) -> Dataset:
        return _self._dm_dataset

    def format(self) -> str:
        return self._format

    def validate(self, task: str):
        if task not in self._val_reports:
            validators = {
                "classification": ClassificationValidator,
                "detection": DetectionValidator,
                "segmentation": SegmentationValidator,
            }
            validator = validators.get(task, ClassificationValidator)()
            reports = validator.validate(self._dm_dataset)
            self._val_reports[task] = reports
        return self._val_reports[task]

    def aggregate(_self, from_subsets, to_subset) -> Dataset:
        _self._dm_dataset = HLOps.aggregate(
            _self._dm_dataset, from_subsets=from_subsets, to_subset=to_subset
        )
        _self._val_reports = {}
        return _self._dm_dataset

    def transform(_self, method, **kwargs):
        _self._dm_dataset = _self._dm_dataset.transform(method, **kwargs)
        _self._val_reports = {}
        return _self._dm_dataset

    def export(_self, save_dir: str, format: str, **kwargs):
        _self._dm_dataset.export(save_dir=save_dir, format=format, **kwargs)
