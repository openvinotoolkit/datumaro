# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict

import pandas as pd

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.extractor import Transform

from .algorithm.algorithm import Algorithm, SamplingMethod


class Sampler(Transform, CliPlugin):
    r"""
    Sampler that analyzes model inference results on the dataset |n
    and picks the best sample for training.|n
    |n
    Notes:|n
    - Each image's inference result must contain the probability for
    all classes.|n
    - Requesting a sample larger than the number of all images will
    return all images.|n
    |n
    Example: select the most relevant data subset of 20 images |n
    |s|sbased on model certainty, put the result into 'sample' subset
    |s|sand put all the rest into 'unsampled' subset, use 'train' subset |n
    |s|sas input. |n
    |s|s%(prog)s \ |n
    |s|s|s|s--algorithm entropy \ |n
    |s|s|s|s--subset_name train \ |n
    |s|s|s|s--sample_name sample \ |n
    |s|s|s|s--unsampled_name unsampled \ |n
    |s|s|s|s--sampling_method topk -k 20
    """

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument("-k", "--count", type=int, required=True,
            help="Number of items to sample")
        parser.add_argument("-a", "--algorithm",
            default=Algorithm.entropy.name,
            choices=[t.name for t in Algorithm],
            help="Sampling algorithm (one of {}; default: %(default)s)".format(
                ', '.join(t.name for t in Algorithm)))
        parser.add_argument("-i", "--input_subset", default=None,
            help="Subset name to select sample from (default: %(default)s)")
        parser.add_argument("-o", "--sampled_subset", default="sample",
            help="Subset name to put sampled data to (default: %(default)s)")
        parser.add_argument("-u", "--unsampled_subset", default="unsampled",
            help="Subset name to put the rest data to (default: %(default)s)")
        parser.add_argument("-m", "--sampling_method",
            default=SamplingMethod.topk.name,
            choices=[t.name for t in SamplingMethod],
            help="Sampling method (one of {}; default: %(default)s)".format(
                ', '.join(t.name for t in SamplingMethod)))
        parser.add_argument("-d", "--output_file",
            help="A .csv file path to dump sampling results file path")
        return parser

    def __init__(self, extractor, algorithm, input_subset, sampled_subset,
            unsampled_subset, sampling_method, count, output_file):
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
        self.input_subset = input_subset
        self.sampled_subset = sampled_subset
        self.unsampled_subset = unsampled_subset
        self.algorithm = algorithm
        self.sampling_method = sampling_method
        self.count = count
        self.output_file = output_file

        # Use the --output_file option to save the sample list as a csv file
        if output_file and not output_file.endswith(".csv"):
            raise ValueError("The output file must have the '.csv' extension")

    @staticmethod
    def _load_inference_from_subset(extractor, subset_name):
        # 1. Get Dataset from subset name
        if subset_name in extractor.subsets():
            subset = extractor.get_subset(subset_name)
        else:
            raise Exception(f"Unknown subset '{subset_name}'")

        data_df = defaultdict(list)
        infer_df = defaultdict(list)

        # 2. Fill the data_df and infer_df to fit the sampler algorithm
        # input format.
        for item in subset:
            data_df['ImageID'].append(item.id)

            if not item.has_image or item.image.size is None:
                raise Exception(f"Item {item.id} does not have image info")

            width, height = item.image.size
            data_df['Width'].append(width)
            data_df['Height'].append(height)
            data_df['ImagePath'].append(item.image.path)

            if not item.annotations:
                raise Exception(f"Item {item.id} does not have annotations")

            for annotation in item.annotations:
                if 'scores' not in annotation.attributes:
                    raise Exception(f"Item {item.id} - an annotation "
                        "does not have 'scores' attribute")
                probs = annotation.attributes['scores']

                infer_df['ImageID'].append(item.id)

                for prob_idx, prob in enumerate(probs):
                    infer_df[f"ClassProbability{prob_idx+1}"].append(prob)

        data_df = pd.DataFrame(data_df)
        infer_df = pd.DataFrame(infer_df)

        return data_df, infer_df

    @staticmethod
    def _calculate_uncertainty(algorithm, data, inference):
        # Checking and creating algorithms
        if algorithm == Algorithm.entropy.name:
            from .algorithm.entropy import SampleEntropy

            # Data delivery, uncertainty score calculations also proceed.
            sampler = SampleEntropy(data, inference)
        else:
            raise Exception(f"Unknown algorithm '{algorithm}', available "
                f"algorithms: {[a.name for a in Algorithm]}")
        return sampler

    def _get_sample_subset(self, image):
        if image.subset == self.input_subset:
            # 1. Returns the sample subset if the id belongs to samples.
            if image.id in self.sample_id:
                return self.sampled_subset
            else:
                return self.unsampled_subset
        else:
            # 2. Returns the existing subset name if it is not a sample
            return image.subset

    def __iter__(self):
        # Import data into a subset name and convert it
        # to a format that will be used in the sampler algorithm with the inference result.
        data_df, infer_df = self._load_inference_from_subset(
            self._extractor, self.input_subset)

        sampler = self._calculate_uncertainty(self.algorithm, data_df, infer_df)
        self.result = sampler.get_sample(method=self.sampling_method,
            k=self.count)

        if self.output_file is not None:
            self.result.to_csv(self.output_file, index=False)

        self.sample_id = self.result['ImageID'].to_list()

        # Transform properties for each data
        for item in self._extractor:
            # After checking whether each item belongs to a sample,
            # rename the subset
            yield self.wrap_item(item, subset=self._get_sample_subset(item))
