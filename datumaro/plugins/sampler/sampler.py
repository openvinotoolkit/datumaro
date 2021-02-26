# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import pandas as pd
from collections import defaultdict
from .algorithm.algorithm import SamplingMethod

from datumaro.components.extractor import Transform, DEFAULT_SUBSET_NAME
from datumaro.components.cli_plugin import CliPlugin


class Sampler(Transform, CliPlugin):
    """
    Sampler that analyzes the inference result of the dataset |n
    and picks the best sample for training.|n
    |n
    Notes:|n
    - Each image's inference result must contain the probability for all classes.|n
    - Requesting a sample larger than the number of all images will return all images.|n
    |n
    Example:|n
    |s|s%(prog)s -algo entropy -subset_name train -sample_name sample -unsampled_name unsampled -m topk -k 20
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-a",
            "--algorithm",
            type=str,
            default="entropy",
            choices=["entropy"],
            help="Select Algorithm ['entropy']",
        )
        parser.add_argument(
            "-subset_name",
            "--subset_name",
            type=str,
            help="Subset name to select sample",
        )
        parser.add_argument(
            "-sample_name",
            "--sampled_name",
            type=str,
            default="sampled_set",
            help="sampled data subset name",
        )
        parser.add_argument(
            "-unsample_name",
            "--unsampled_name",
            type=str,
            default="unsampled_set",
            help="unsampled data subset name name",
        )
        parser.add_argument(
            "-m",
            "--sampling_method",
            type=str,
            default="topk",
            choices=[t.name for t in SamplingMethod],
            help=f"Method of sampling, example: {[t.name for t in SamplingMethod]}",
        )
        parser.add_argument("-k", "--num_sample", type=int, help="Num of sample")
        parser.add_argument(
            "-o",
            "--output_file",
            type=str,
            default=None,
            help="Output Sample file path, The extension of the file must end with .csv",
        )
        return parser

    def __init__(
        self,
        extractor,
        algorithm,
        subset_name,
        sampled_name,
        unsampled_name,
        sampling_method,
        num_sample,
        output_file,
    ):
        """
        Parameters
        ----------
        extractor : Extractor, Dataset
        algorithm : str
            Specifying the algorithm to calculate the uncertainty
            for sample selection. default: 'entropy'
        subset_name : str
            The name of the subset to which you want to select a sample.
        sample_name : str
            Subset name of the selected sample, default: 'sample'
        sampling_method : str
            Method of sampling, 'topk' or 'lowk' or 'randk'
        num_sample : int
            Number of samples extracted
        output_file : str
            Path of sampler result, Use when user want to save results
        """
        super().__init__(extractor)

        # Get Parameters
        self.subset_name = subset_name
        self.sampled_name = sampled_name
        self.unsampled_name = unsampled_name
        self.algorithm = algorithm
        self.sampling_method = sampling_method
        self.num_sample = num_sample
        self.output_file = output_file

        # optional. Use the --output_file option to save the sample list as a csv file.
        if output_file is not None:
            if output_file.split(".")[-1] != ".csv":
                msg = f"Invalid extension, The extension of the file must end with .csv"
                raise Exception(msg)

    @staticmethod
    def _load_inference_from_subset(extractor, subset_name):
        # 1. Get Dataset from subset name
        if subset_name in extractor.subsets().keys():
            subset = extractor.get_subset(subset_name)
        else:
            msg = f"Not Found subset '{subset_name}'"
            raise Exception(msg)

        data_df = defaultdict(list)
        infer_df = defaultdict(list)

        # 2. Fill the data_df and infer_df to fit the sampler algorithm input format.
        for data in subset:
            data_df["ImageID"].append(data.id)

            if not data.has_image or data.image.size is None:
                msg = "Invalid data, the image file is not available"
                raise Exception(msg)

            width, height = data.image.size
            data_df["Width"].append(width)
            data_df["Height"].append(height)
            data_df["ImagePath"].append(data.image.path)

            if not data.annotations:
                msg = f"Invalid data, data.annotations is empty"
                raise Exception(msg)

            for annotation in data.annotations:
                if "score" not in annotation.attributes:
                    msg = f"Invalid data, probability score is None"
                    raise Exception(msg)
                probs = annotation.attributes["score"]

                infer_df["ImageID"].append(data.id)

                for prob_idx, prob in enumerate(probs):
                    infer_df[f"ClassProbability{prob_idx+1}"].append(prob)

        data_df = pd.DataFrame(data_df)
        infer_df = pd.DataFrame(infer_df)

        return data_df, infer_df

    @staticmethod
    def _calculate_uncertainty(algorithm, data, inference):
        # Checking and creating algorithms
        algorithms = ["entropy"]
        if algorithm == "entropy":
            from .algorithm.entropy import SampleEntropy

            # Data delivery, uncertainty score calculations also proceed.
            sampler = SampleEntropy(data, inference)
        else:
            msg = (
                f"Not Found algorithm '{algorithm}', available algorithms: {algorithms}"
            )
            raise Exception(msg)
        return sampler

    def _check_sample(self, image):
        # The function that determines the subset name of the data.
        if image.subset:
            if image.subset == self.subset_name:
                # 1. Returns the sample subset if the id of that image belongs to the list of samples.
                if image.id in self.sample_id:
                    return self.sampled_name
                else:
                    return self.unsampled_name
            else:
                # 2. Returns the existing subset name if it is not a sample
                return image.subset
        else:
            return DEFAULT_SUBSET_NAME

    def __iter__(self):
        # Import data into a subset name and convert it
        # to a format that will be used in the sampler algorithm with the inference result.
        data_df, infer_df = self._load_inference_from_subset(
            self._extractor, self.subset_name
        )

        # Transfer the data to sampler algorithm to calculate uncertainty & get sample list
        sampler = self._calculate_uncertainty(self.algorithm, data_df, infer_df)
        self.result = sampler.get_sample(method=self.sampling_method, k=self.num_sample)

        if self.output_file is not None:
            self.result.to_csv(self.output_file, index=False)

        self.sample_id = self.result["ImageID"].to_list()

        # Transform properties for each data
        for item in self._extractor:
            # After checking whether each item belongs to a sample, rename the subset.
            yield self.wrap_item(item, subset=self._check_sample(item))
