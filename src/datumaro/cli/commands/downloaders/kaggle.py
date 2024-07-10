# Copyright (C) 2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import json
import logging as log
import os
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

from datumaro.cli.commands.require_project.modification import create, import_

from ...util.errors import CliException
from . import IDatasetDownloader

with open(Path(__file__).parent / "kaggle_formats.json") as f:
    _SUPPORTED_DATASETS = json.load(f)


def make_all_paths_absolute(d: Dict, root: str = "."):
    for k, v in d.items():
        if isinstance(v, dict):
            make_all_paths_absolute(v, root)
        if isinstance(v, str):
            relpath = Path(root) / v
            if relpath.exists():
                d[k] = str(relpath.resolve())


KAGGLE_API_KEY_EXISTS = bool(os.environ.get("KAGGLE_KEY")) or os.path.exists(
    os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
)


class KaggleDatasetDownloader(IDatasetDownloader):
    @classmethod
    def download(
        cls,
        dataset_id,
        dst_dir=None,
        overwrite=False,
        output_format=None,
        subset=None,
        extra_args=None,
    ):
        try:
            import kaggle
        except ImportError:
            raise CliException(
                "Kaggle API is not installed. To install it, run `pip install kaggle`."
            )

        import_kwargs = _SUPPORTED_DATASETS.get(dataset_id, {})
        if not import_kwargs:
            if not extra_args:
                raise CliException(
                    f"Dataset {dataset_id} has no datumaro-compatible implementation.\n"
                    "Please specify the format and constructor arguments explicitly:\n"
                    "'-- --format=<format> --arg1=<arg1> --arg2=<arg2> ...'"
                )
        else:
            log.info(f"{dataset_id} is supported. Settings:\n{import_kwargs}")
        if output_format:
            log.info(f"Overriding the format with {output_format}...")
            import_kwargs["format"] = output_format
        if "subsets" in import_kwargs:
            if not subset or subset not in import_kwargs["subsets"]:
                raise CliException(
                    f"Please specify the subset. Options : {[k for k in import_kwargs['subsets']]}"
                )
            log.info(f"Getting subset {subset}...")
            import_kwargs = import_kwargs["subsets"][subset]
        if extra_args:
            parsed = import_.build_parser().parse(extra_args)
            format = parsed.format
        else:
            format = import_kwargs.pop("format")

        with TemporaryDirectory() as tmp_dir:
            kaggle.api.dataset_download_cli(dataset_id, path=tmp_dir, force=overwrite, unzip=True)
            create.create_command(Namespace(dst_dir=dst_dir))
            make_all_paths_absolute(import_kwargs, tmp_dir)
            if not extra_args:
                extra_args = {}
                for k, v in import_kwargs.items():
                    if isinstance(v, (dict, list, tuple)):
                        extra_args[k] = json.dumps(v)
                    else:
                        extra_args[k] = str(v)
                extra_args = [f"--{k}={v}" for k, v in extra_args.items()]
            source_name = kaggle.api.split_dataset_string(dataset_id)[1]
            import_.import_command(
                Namespace(
                    _positionals=[],
                    url=tmp_dir,
                    project_dir=dst_dir,
                    format=format,
                    extra_args=extra_args,
                    name=source_name,
                    rpath=None,
                    no_check=False,
                )
            )
            log.info("Dataset downloaded successfully.")

    @classmethod
    def describe(cls, report_format="txt", report_file=None) -> None:
        file = report_file if report_file else None
        if report_format == "txt":
            print(cls.get_command_description(), file=file)

    @classmethod
    def get_command_description(cls) -> str:
        return f"""
Supported datasets: {os.linesep.join(_SUPPORTED_DATASETS)}|n
|n
Examples:|n
- Download the face mask detection dataset:|n
|s|s%(prog)s -i andrewmvd/face-mask-detection"""

    @classmethod
    def describe_command_description(cls):
        return """More detailed
        information can be found in the Kaggle datasets catalog:
        <https://www.kaggle.com/datasets>."""
