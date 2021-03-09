from collections import defaultdict
from unittest import TestCase, skipIf

from datumaro.components.project import Dataset
from datumaro.components.extractor import (
    DatasetItem,
    Label,
    LabelCategories,
    AnnotationType,
)
from datumaro.util.image import Image

import csv

try:
    import pandas as pd
    from datumaro.plugins.sampler.sampler import Sampler
    from datumaro.plugins.sampler.algorithm.entropy import SampleEntropy as entropy
    has_libs = True
except ImportError:
    has_libs = False


@skipIf(not has_libs, "pandas library is not available")
class SamplerTest(TestCase):
    @staticmethod
    def _get_probs(out_range=False):
        probs = []
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

    def _generate_classification_dataset(self, config, subset=None,
            empty_confidences=False, out_range=False, no_attr=False, no_img=False):
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
                confidences = probs[idx]
                idx += 1
                if empty_confidences:
                    confidences = []
                attr = {"confidences": confidences}
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
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config, ["train"])
        num_pre_train_subset = len(source.get_subset("train"))

        num_sample = 5

        with self.subTest("Top-K method"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            topk_expected_result = [1, 4, 9, 10, 26]
            topk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(topk_result), topk_expected_result)

        with self.subTest("Low-K method"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="lowk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            lowk_expected_result = [2, 6, 14, 21, 23]
            lowk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(lowk_result), lowk_expected_result)

        with self.subTest("Rand-K method"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="randk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

        with self.subTest("Mix-K method"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="mixk",
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            mixk_expected_result = [2, 4, 10, 23, 26]
            mixk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="mixk",
                count=6,
                output_file=None,
            )
            self.assertEqual(6, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )
            mixk_expected_result = [2, 4, 6, 10, 23, 26]
            mixk_result = list(map(int, result.result["ImageID"].to_list()))
            self.assertEqual(sorted(mixk_result), mixk_expected_result)

        with self.subTest("Randtop-K method"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method="randtopk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(num_sample, len(result.get_subset("sample")))
            self.assertEqual(
                len(result.get_subset("unsampled")),
                num_pre_train_subset - len(result.get_subset("sample")),
            )

    def test_sampler_gives_error(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }
        num_sample = 5

        source = self._generate_classification_dataset(config)

        with self.subTest("Not found"):
            with self.assertRaisesRegex(Exception, "Unknown subset"):
                subset = "hello"
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset=subset,
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Unknown algorithm"):
                algorithm = "hello"
                result = Sampler(
                    source,
                    algorithm=algorithm,
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Unknown sampling method"):
                sampling_method = "hello"
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method=sampling_method,
                    count=num_sample,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Invalid Value"):
            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = 0
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = -1
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "Invalid value"):
                k = "string"
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=k,
                    output_file=None,
                )
                result = iter(result)
                next(result)

            with self.assertRaisesRegex(Exception, "extension"):
                output_file = "string.xml"
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=num_sample,
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
                        probs = annotation.attributes["confidences"]
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
                        probs = annotation.attributes["confidences"]

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
            with self.assertRaisesRegex(Exception, "Unknown subset"):
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Dataset without confidences (Probability)"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, empty_confidences=True)
            with self.assertRaisesRegex(Exception, "ClassProbability"):
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("Out of range, probability (Less than 0 or more than 1)"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(
                config, empty_confidences=False, out_range=True
            )
            with self.assertRaisesRegex(Exception, "Invalid data"):
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No confidences Attribute Data"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, no_attr=True)
            with self.assertRaisesRegex(Exception, "does not have 'confidences'"):
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

        with self.subTest("No Image Data"):
            config = {
                "label1": 10,
                "label2": 10,
                "label3": 10,
            }

            source = self._generate_classification_dataset(config, no_img=True)
            with self.assertRaisesRegex(Exception, "does not have image info"):
                result = Sampler(
                    source,
                    algorithm="entropy",
                    input_subset="train",
                    sampled_subset="sample",
                    unsampled_subset="unsampled",
                    sampling_method="topk",
                    count=5,
                    output_file=None,
                )
                result = iter(result)
                next(result)

    def test_sampler_number_of_samples(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)
        num_pre_train_subset = len(source.get_subset("train"))

        with self.subTest("k > num of data with top-k"):
            num_sample = 500
            sampling_method = "topk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with low-k"):
            num_sample = 500
            sampling_method = "lowk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with rand-k"):
            num_sample = 500
            sampling_method = "randk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with mix-k"):
            num_sample = 500
            sampling_method = "mixk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k > num of data with randtop-k"):
            num_sample = 500
            sampling_method = "randtopk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with top-k"):
            num_sample = 10
            sampling_method = "topk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with low-k"):
            num_sample = 10
            sampling_method = "lowk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with rand-k"):
            num_sample = 10
            sampling_method = "randk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with mix-k"):
            num_sample = 10
            sampling_method = "mixk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

        with self.subTest("k == num of data with randtop-k"):
            num_sample = 10
            sampling_method = "randtopk"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(num_pre_train_subset, len(result.get_subset("sample")))

            num_sample = 9

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample",
                unsampled_subset="unsampled",
                sampling_method=sampling_method,
                count=num_sample,
                output_file=None,
            )
            self.assertEqual(len(result.get_subset("sample")), 9)

    def test_sampler_accumulated_sampling(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 3
            sample_subset_name = "sample"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 2
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 3
            )

        with self.subTest("Same Subset, 2, 3, 4 sampling"):
            sample_subset_name = "sample"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 5)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 5)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 9)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 9)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 3
            sample_subset_name = "sample"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset=sample_subset_name,
                unsampled_subset="val",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 2)
            self.assertEqual(
                len(result.get_subset("val")), num_pre_val_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset=sample_subset_name,
                unsampled_subset="test",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), num_sample * 3)
            self.assertEqual(
                len(result.get_subset("test")), num_pre_test_subset - num_sample
            )

        with self.subTest("Different Subset, 2, 3, 4 sampling"):
            sample_subset_name = "sample"

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset=sample_subset_name,
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset=sample_subset_name,
                unsampled_subset="val",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 5)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 3)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset=sample_subset_name,
                unsampled_subset="test",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample")), 9)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 4)

    def test_sampler_unaccumulated_sampling(self):
        config = {
            "label1": 10,
            "label2": 10,
            "label3": 10,
        }

        source = self._generate_classification_dataset(config)

        num_pre_train_subset = len(source.get_subset("train"))
        num_pre_val_subset = len(source.get_subset("val"))
        num_pre_test_subset = len(source.get_subset("test"))

        with self.subTest("Same Subset, Same number of datas 3times"):
            num_sample = 3

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample2",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 2
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample3",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample * 3
            )

        with self.subTest("Same Subset, 2, 3, 4 sampling"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample2",
                unsampled_subset="train",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 5)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample3",
                unsampled_subset="train",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("sample3")), 4)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 9)

        with self.subTest("Different Subset, Same number of datas 3times"):
            num_sample = 3

            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(
                len(result.get_subset("train")), num_pre_train_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset="sample2",
                unsampled_subset="val",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(
                len(result.get_subset("val")), num_pre_val_subset - num_sample
            )

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset="sample3",
                unsampled_subset="test",
                sampling_method="topk",
                count=num_sample,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), num_sample)
            self.assertEqual(len(result.get_subset("sample2")), num_sample)
            self.assertEqual(len(result.get_subset("sample3")), num_sample)
            self.assertEqual(
                len(result.get_subset("test")), num_pre_test_subset - num_sample
            )

        with self.subTest("Different Subset, 2, 3, 4 sampling"):
            result = Sampler(
                source,
                algorithm="entropy",
                input_subset="train",
                sampled_subset="sample1",
                unsampled_subset="train",
                sampling_method="topk",
                count=2,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("train")), num_pre_train_subset - 2)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="val",
                sampled_subset="sample2",
                unsampled_subset="val",
                sampling_method="topk",
                count=3,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("val")), num_pre_val_subset - 3)

            result = Sampler(
                result,
                algorithm="entropy",
                input_subset="test",
                sampled_subset="sample3",
                unsampled_subset="test",
                sampling_method="topk",
                count=4,
                output_file=None,
            )

            self.assertEqual(len(result.get_subset("sample1")), 2)
            self.assertEqual(len(result.get_subset("sample2")), 3)
            self.assertEqual(len(result.get_subset("sample3")), 4)
            self.assertEqual(len(result.get_subset("test")), num_pre_test_subset - 4)

    def test_sampler_parser(self):
        from argparse import ArgumentParser

        assert isinstance(Sampler.build_cmdline_parser(), ArgumentParser)
