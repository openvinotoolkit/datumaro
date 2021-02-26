from collections import defaultdict
from unittest import TestCase

from datumaro.components.project import Dataset
from datumaro.components.extractor import (
    DatasetItem,
    Label,
    LabelCategories,
    AnnotationType,
)
from datumaro.util.image import Image

import csv
import pandas as pd

import datumaro.plugins.sampler.sampler as sampler
from datumaro.plugins.sampler.algorithm.entropy import SampleEntropy as entropy


class SamplerTest(TestCase):
    @staticmethod
    def _get_probs(out_range=False):
        probs = []
        # data length is 500
        inference_file = "tests/assets/sampler/inference.csv"
        with open(inference_file) as csv_file:
            csv_reader = csv.reader(csv_file)
            col = 0
            for row in csv_reader:
                if col == 0:
                    col += 1
                    continue
                else:
                    if out_range:
                        probs.append(list(map(lambda x: -float(x), row[1:4])))
                    else:
                        probs.append(list(map(float, row[1:4])))
        return probs

    def _generate_classification_dataset(
        self,
        config,
        subset=None,
        empty_score=False,
        out_range=False,
        no_attr=False,
        no_img=False,
    ):

        probs = self._get_probs(out_range)
        if subset is None:
            self.subset = ["train", "val", "test"]
        else:
            self.subset = subset

        iterable = []
        label_cat = LabelCategories()
        idx = 0
        for label_id, label in enumerate(config.keys()):
            num_item = config[label]
            label_cat.add(label, attributes=None)
            for _ in range(num_item):
                score = probs[idx]
                idx += 1
                if empty_score:
                    score = []
                attr = {"score": score}
                if no_attr:
                    attr = {}
                img = Image(path=f"test/dataset/{idx}.jpg", size=(90, 90))
                if no_img:
                    img = None
                iterable.append(
                    DatasetItem(
                        idx,
                        subset=self.subset[idx % len(self.subset)],
                        annotations=[
                            Label(
                                label_id,
                                attributes=attr,
                            )
                        ],
                        image=img,
                    )
                )
        categories = {AnnotationType.label: label_cat}
        dataset = Dataset.from_iterable(iterable, categories)
        return dataset

    def test_sampler_get_sample_classification(self):
        config = {
            "label1": 100,
            "label2": 200,
            "label3": 200,
        }

        source = self._generate_classification_dataset(config, ["train"])
        num_pre_train_subset = len(source.get_subset("train"))

        num_sample = 20

        with self.subTest("Top-K method"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

            topk_expected_result = [
                10,
                39,
                48,
                58,
                73,
                94,
                157,
                178,
                185,
                187,
                205,
                327,
                358,
                379,
                418,
                421,
                448,
                450,
                451,
                484,
            ]
            topk_result = list(map(float, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(topk_result), topk_expected_result)

        with self.subTest("Low-K method"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="lowk",
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

            lowk_expected_result = [
                33,
                121,
                124,
                180,
                194,
                217,
                250,
                276,
                289,
                298,
                313,
                354,
                375,
                388,
                395,
                397,
                428,
                466,
                477,
                488,
            ]
            lowk_result = list(map(float, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(lowk_result), lowk_expected_result)

        with self.subTest("Rand-K method"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="randk",
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

        with self.subTest("Mix-K method"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="mixk",
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

            mixk_expected_result = [
                33,
                39,
                48,
                58,
                121,
                124,
                187,
                205,
                276,
                289,
                298,
                313,
                327,
                375,
                379,
                397,
                448,
                451,
                466,
                484,
            ]
            mixk_result = list(map(float, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="mixk",
                num_sample=21,
                output_file=None,
            )
            self.assertEqual(21, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            mixk_expected_result = [
                10,
                33,
                39,
                48,
                58,
                121,
                124,
                187,
                205,
                276,
                289,
                298,
                313,
                327,
                375,
                379,
                397,
                448,
                451,
                466,
                484,
            ]
            mixk_result = list(map(float, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

        with self.subTest("Randtop-K method"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method="randtopk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

    def test_sampler_gives_error(self):
        config = {
            "label1": 100,
            "label2": 100,
            "label3": 100,
        }

        source = self._generate_classification_dataset(config)

        with self.subTest("Not found"):
            with self.assertRaisesRegex(Exception, "Not Found subset"):
                subset = "hello"
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name=subset,
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Not Found algorithm"):
                algorithm = "hello"
                result = sampler.Sampler(
                    source,
                    algorithm=algorithm,
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Not Found method"):
                sampling_method = "hello"
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method=sampling_method,
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Invalid Value"):
            with self.assertRaisesRegex(Exception, "Invalid number"):
                k = 0
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid number"):
                k = -1
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = "string"
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid extension"):
                output_file = "string.xml"
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=output_file,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid extension"):
                output_file = "string"
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=output_file,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(
                Exception, "Invalid Data, ImageID not found in data"
            ):
                sub = source.get_subset("train")

                data_df = defaultdict(list)
                infer_df = defaultdict(list)

                for data in sub:
                    width, height = data.image.size
                    data_df["Width"].append(width)
                    data_df["Height"].append(height)
                    data_df["ImagePath"].append(data.image.path)

                    for annotation in data.annotations:
                        probs = annotation.attributes["score"]
                        infer_df["ImageID"].append(data.id)

                        for prob_idx, prob in enumerate(probs):
                            infer_df[f"ClassProbability{prob_idx+1}"].append(prob)

                data_df = pd.DataFrame(data_df)
                infer_df = pd.DataFrame(infer_df)

                entropy(data_df, infer_df)

            with self.assertRaisesRegex(
                Exception, "Invalid Data, ImageID not found in inference"
            ):
                sub = source.get_subset("train")

                data_df = defaultdict(list)
                infer_df = defaultdict(list)

                for data in sub:
                    width, height = data.image.size
                    data_df["ImageID"].append(data.id)
                    data_df["Width"].append(width)
                    data_df["Height"].append(height)
                    data_df["ImagePath"].append(data.image.path)

                    for annotation in data.annotations:
                        probs = annotation.attributes["score"]

                        for prob_idx, prob in enumerate(probs):
                            infer_df[f"ClassProbability{prob_idx+1}"].append(prob)

                data_df = pd.DataFrame(data_df)
                infer_df = pd.DataFrame(infer_df)

                entropy(data_df, infer_df)

    def test_sampler_get_invalid_data(self):
        with self.subTest("empty dataset"):
            config = {
                "label1": 0,
                "label2": 0,
                "label3": 0,
            }

            source = self._generate_classification_dataset(config)
            with self.assertRaisesRegex(Exception, "Not Found"):
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Dataset without Score (Probability)"):
            config = {
                "label1": 100,
                "label2": 100,
                "label3": 100,
            }

            source = self._generate_classification_dataset(config, empty_score=True)
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Out of range, probability (Less than 0 or more than 1)"):
            config = {
                "label1": 100,
                "label2": 100,
                "label3": 100,
            }

            source = self._generate_classification_dataset(
                config, empty_score=False, out_range=True
            )
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No Score Attribute Data"):
            config = {
                "label1": 100,
                "label2": 100,
                "label3": 100,
            }

            source = self._generate_classification_dataset(config, no_attr=True)
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No Image Data"):
            config = {
                "label1": 100,
                "label2": 100,
                "label3": 100,
            }

            source = self._generate_classification_dataset(config, no_img=True)
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = sampler.Sampler(
                    source,
                    algorithm="entropy",
                    subset_name="train",
                    sampled_name="sample",
                    unsampled_name="unsampled",
                    sampling_method="topk",
                    num_sample=20,
                    output_file=None,
                )
                result = iter(result)
                next(result)

    def test_sampler_number_of_samples(self):
        config = {
            "label1": 100,
            "label2": 100,
            "label3": 100,
        }

        source = self._generate_classification_dataset(config)
        num_pre_train_subset = len(source.get_subset("train"))

        with self.subTest("k > num of data with top-k"):
            num_sample = 500
            sampling_method = "topk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with low-k"):
            num_sample = 500
            sampling_method = "lowk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with rand-k"):
            num_sample = 500
            sampling_method = "randk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with mix-k"):
            num_sample = 500
            sampling_method = "mixk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with randtop-k"):
            num_sample = 500
            sampling_method = "randtopk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with top-k"):
            num_sample = 100
            sampling_method = "topk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with low-k"):
            num_sample = 100
            sampling_method = "lowk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with rand-k"):
            num_sample = 100
            sampling_method = "randk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with mix-k"):
            num_sample = 100
            sampling_method = "mixk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with randtop-k"):
            num_sample = 100
            sampling_method = "randtopk"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

            num_sample = 99

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample",
                unsampled_name="unsampled",
                sampling_method=sampling_method,
                num_sample=num_sample,
                output_file=None,
            )
            self.assertEqual(len(result.get_subset("sample")), 99)

    def test_sampler_accumulated_sampling(self):
        config = {
            "label1": 100,
            "label2": 100,
            "label3": 100,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 20
            sample_subset_name = "sample"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 2
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 3
            )

        with self.subTest("Same Subset, 10, 20, 30 sampling"):
            sample_subset_name = "sample"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=10,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 10)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 10)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=20,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 30)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 30)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=30,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 60)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 60)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 20
            sample_subset_name = "sample"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="val",
                sampled_name=sample_subset_name,
                unsampled_name="val",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(
                len(result.get_subset("val")), num_pre_val_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="test",
                sampled_name=sample_subset_name,
                unsampled_name="test",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(
                len(result.get_subset("test")), num_pre_test_subset - num_sample
            )

        with self.subTest("Different Subset, 10, 20, 30 sampling"):
            sample_subset_name = "sample"

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name=sample_subset_name,
                unsampled_name="train",
                sampling_method="topk",
                num_sample=10,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 10)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 10)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="val",
                sampled_name=sample_subset_name,
                unsampled_name="val",
                sampling_method="topk",
                num_sample=20,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 30)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 20)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="test",
                sampled_name=sample_subset_name,
                unsampled_name="test",
                sampling_method="topk",
                num_sample=30,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 60)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 30)

    def test_sampler_unaccumulated_sampling(self):
        config = {
            "label1": 100,
            "label2": 100,
            "label3": 100,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 20

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample1",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample2",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 2
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample3",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 3
            )

        with self.subTest("Same Subset, 10, 20, 30 sampling"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample1",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=10,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 10)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample2",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=20,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("sample2")), 20)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 30)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample3",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=30,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("sample2")), 20)
            self.assertEqual(len(result.get_subset("sample3")), 30)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 60)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 20

            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample1",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="val",
                sampled_name="sample2",
                unsampled_name="val",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(
                len(result.get_subset("val")), num_pre_val_subset - num_sample
            )

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="test",
                sampled_name="sample3",
                unsampled_name="test",
                sampling_method="topk",
                num_sample=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(
                len(result.get_subset("test")), num_pre_test_subset - num_sample
            )

        with self.subTest("Different Subset, 10, 20, 30 sampling"):
            result = sampler.Sampler(
                source,
                algorithm="entropy",
                subset_name="train",
                sampled_name="sample1",
                unsampled_name="train",
                sampling_method="topk",
                num_sample=10,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 10)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="val",
                sampled_name="sample2",
                unsampled_name="val",
                sampling_method="topk",
                num_sample=20,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("sample2")), 20)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 20)

            result = sampler.Sampler(
                result,
                algorithm="entropy",
                subset_name="test",
                sampled_name="sample3",
                unsampled_name="test",
                sampling_method="topk",
                num_sample=30,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 10)
            self.assertEqual(len(result.get_subset("sample2")), 20)
            self.assertEqual(len(result.get_subset("sample3")), 30)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 30)

    def test_sampler_parser(self):
        from argparse import ArgumentParser

        assert isinstance(sampler.Sampler.build_cmdline_parser(), ArgumentParser)
