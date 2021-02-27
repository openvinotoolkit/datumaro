import numpy as np

from unittest import TestCase

from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem, Label, Bbox,
    LabelCategories, AnnotationType)

import datumaro.plugins.splitter as splitter
from datumaro.components.operations import compute_ann_statistics


class SplitterTest(TestCase):
    @staticmethod
    def _get_subset(idx):
        subsets = ["", "a", "b", "", "", "a", "", "b", "", "a"]
        return subsets[idx % len(subsets)]

    def _generate_dataset(self, config):
        # counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        # attr1 = ['attr1', 'attr2']
        # attr2 = ['attr1', 'attr3']
        # config = { "label1": { "attrs": attr1, "counts": counts },
        #            "label2": { "attrs": attr2, "counts": counts }}
        iterable = []
        label_cat = LabelCategories()
        idx = 0
        for label_id, label in enumerate(config.keys()):
            anames = config[label]["attrs"]
            counts = config[label]["counts"]
            label_cat.add(label, attributes=anames)
            if isinstance(counts, dict):
                for attrs, count in counts.items():
                    attributes = dict()
                    if isinstance(attrs, tuple):
                        for aname, value in zip(anames, attrs):
                            attributes[aname] = value
                    else:
                        attributes[anames[0]] = attrs
                    for _ in range(count):
                        idx += 1
                        iterable.append(
                            DatasetItem(idx, subset=self._get_subset(idx),
                                annotations=[
                                    Label(label_id, attributes=attributes)
                                ],
                            )
                        )
            else:
                for _ in range(counts):
                    idx += 1
                    iterable.append(
                        DatasetItem(idx, subset=self._get_subset(idx),
                            annotations=[Label(label_id)])
                    )
        categories = {AnnotationType.label: label_cat}
        dataset = Dataset.from_iterable(iterable, categories)
        return dataset

    def test_split_for_classification_multi_class_no_attr(self):
        config = {
            "label1": {"attrs": None, "counts": 10},
            "label2": {"attrs": None, "counts": 20},
            "label3": {"attrs": None, "counts": 30},
        }
        source = self._generate_dataset(config)

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.ClassificationSplit(source, splits)

        self.assertEqual(42, len(actual.get_subset("train")))
        self.assertEqual(18, len(actual.get_subset("test")))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset("train"))
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(7, dist_train["label1"][0])
        self.assertEqual(14, dist_train["label2"][0])
        self.assertEqual(21, dist_train["label3"][0])

        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset("test"))
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(3, dist_test["label1"][0])
        self.assertEqual(6, dist_test["label2"][0])
        self.assertEqual(9, dist_test["label3"][0])

    def test_split_for_classification_single_class_single_attr(self):
        counts = {0: 10, 1: 20, 2: 30}
        config = {"label": {"attrs": ["attr"], "counts": counts}}
        source = self._generate_dataset(config)

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.ClassificationSplit(source, splits)

        self.assertEqual(42, len(actual.get_subset("train")))
        self.assertEqual(18, len(actual.get_subset("test")))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset("train"))
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(7, attr_train["attr"]["distribution"]["0"][0])
        self.assertEqual(14, attr_train["attr"]["distribution"]["1"][0])
        self.assertEqual(21, attr_train["attr"]["distribution"]["2"][0])

        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset("test"))
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(3, attr_test["attr"]["distribution"]["0"][0])
        self.assertEqual(6, attr_test["attr"]["distribution"]["1"][0])
        self.assertEqual(9, attr_test["attr"]["distribution"]["2"][0])

    def test_split_for_classification_single_class_multi_attr(self):
        counts = {
            (0, 0): 20,
            (0, 1): 20,
            (0, 2): 30,
            (1, 0): 20,
            (1, 1): 10,
            (1, 2): 20,
        }
        attrs = ["attr1", "attr2"]
        config = {"label": {"attrs": attrs, "counts": counts}}
        source = self._generate_dataset(config)

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.ClassificationSplit(source, splits)

        self.assertEqual(84, len(actual.get_subset("train")))
        self.assertEqual(36, len(actual.get_subset("test")))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset("train"))
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(49, attr_train["attr1"]["distribution"]["0"][0])
        self.assertEqual(35, attr_train["attr1"]["distribution"]["1"][0])
        self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])
        self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])
        self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])

        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset("test"))
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(21, attr_test["attr1"]["distribution"]["0"][0])
        self.assertEqual(15, attr_test["attr1"]["distribution"]["1"][0])
        self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])
        self.assertEqual(9, attr_test["attr2"]["distribution"]["1"][0])
        self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])

    def test_split_for_classification_multi_label_with_attr(self):
        counts = {
            (0, 0): 20,
            (0, 1): 20,
            (0, 2): 30,
            (1, 0): 20,
            (1, 1): 10,
            (1, 2): 20,
        }
        attr1 = ["attr1", "attr2"]
        attr2 = ["attr1", "attr3"]
        config = {
            "label1": {"attrs": attr1, "counts": counts},
            "label2": {"attrs": attr2, "counts": counts},
        }
        source = self._generate_dataset(config)

        splits = [("train", 0.7), ("test", 0.3)]
        actual = splitter.ClassificationSplit(source, splits)

        train = actual.get_subset("train")
        test = actual.get_subset("test")
        self.assertEqual(168, len(train))
        self.assertEqual(72, len(test))

        # check stats for train
        stat_train = compute_ann_statistics(train)
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(84, dist_train["label1"][0])
        self.assertEqual(84, dist_train["label2"][0])
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(49 * 2, attr_train["attr1"]["distribution"]["0"][0])
        self.assertEqual(35 * 2, attr_train["attr1"]["distribution"]["1"][0])
        self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])
        self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])
        self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])
        self.assertEqual(28, attr_train["attr3"]["distribution"]["0"][0])
        self.assertEqual(21, attr_train["attr3"]["distribution"]["1"][0])
        self.assertEqual(35, attr_train["attr3"]["distribution"]["2"][0])

        # check stats for test
        stat_test = compute_ann_statistics(test)
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(36, dist_test["label1"][0])
        self.assertEqual(36, dist_test["label2"][0])
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(21 * 2, attr_test["attr1"]["distribution"]["0"][0])
        self.assertEqual(15 * 2, attr_test["attr1"]["distribution"]["1"][0])
        self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])
        self.assertEqual(9, attr_test["attr2"]["distribution"]["1"][0])
        self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])
        self.assertEqual(12, attr_test["attr3"]["distribution"]["0"][0])
        self.assertEqual(9, attr_test["attr3"]["distribution"]["1"][0])
        self.assertEqual(15, attr_test["attr3"]["distribution"]["2"][0])

        # random seed test
        r1 = splitter.ClassificationSplit(source, splits, seed=1234)
        r2 = splitter.ClassificationSplit(source, splits, seed=1234)
        r3 = splitter.ClassificationSplit(source, splits, seed=4321)
        self.assertEqual(
            list(r1.get_subset("test")), list(r2.get_subset("test"))
        )
        self.assertNotEqual(
            list(r1.get_subset("test")), list(r3.get_subset("test"))
        )

    def test_split_for_classification_gives_error(self):
        with self.subTest("no label"):
            source = Dataset.from_iterable([
                DatasetItem(1, annotations=[]),
                DatasetItem(2, annotations=[]),
            ], categories=["a", "b", "c"])

            with self.assertRaisesRegex(Exception, "exactly one is expected"):
                splits = [("train", 0.7), ("test", 0.3)]
                actual = splitter.ClassificationSplit(source, splits)
                len(actual.get_subset("train"))

        with self.subTest("multi label"):
            source = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0), Label(1)]),
                DatasetItem(2, annotations=[Label(0), Label(2)]),
            ], categories=["a", "b", "c"])

            with self.assertRaisesRegex(Exception, "exactly one is expected"):
                splits = [("train", 0.7), ("test", 0.3)]
                splitter.ClassificationSplit(source, splits)
                len(actual.get_subset("train"))

        source = Dataset.from_iterable([
            DatasetItem(1, annotations=[Label(0)]),
            DatasetItem(2, annotations=[Label(1)]),
        ], categories=["a", "b", "c"])

        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("test", 1.5)]
                splitter.ClassificationSplit(source, splits)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                splitter.ClassificationSplit(source, splits)

        with self.subTest("wrong subset name"):
            with self.assertRaisesRegex(Exception, "Subset name"):
                splits = [("train_", 0.5), ("val", 0.2), ("test", 0.3)]
                splitter.ClassificationSplit(source, splits)

    def test_split_for_reidentification_with_label(self):
        counts = {}
        config = dict()
        for i in range(10):
            label = "label%d" % i
            count = (i % 3 + 1) * 7
            counts[label] = count
            config[label] = {"attrs": None, "counts": count}
        source = self._generate_dataset(config)

        splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
        query = 0.4 / 0.7
        actual = splitter.ReidentificationSplit(source, splits, query)

        stats = dict()
        for sname in ["train", "val", "test-query", "test-gallery"]:
            subset = actual.get_subset(sname)
            stat = compute_ann_statistics(subset)["annotations"]["labels"]
            stats[sname] = stat

        self.assertEqual(65, stats["train"]["count"])
        self.assertEqual(26, stats["val"]["count"])
        self.assertEqual(18, stats["test-gallery"]["count"])
        self.assertEqual(24, stats["test-query"]["count"])

        def _get_values_present(stat):
            values_present = []
            for label, dist in stat["distribution"].items():
                if dist[0] > 0:
                    values_present.append(label)
            return set(values_present)

        train_ids = _get_values_present(stats["train"])
        self.assertEqual(7, len(train_ids))
        self.assertEqual(train_ids, _get_values_present(stats["val"]))

        trainval = stats["train"]["count"] + stats["val"]["count"]
        self.assertEqual(int(trainval * 0.5 / 0.7), stats["train"]["count"])
        self.assertEqual(int(trainval * 0.2 / 0.7), stats["val"]["count"])

        dist_train = stats["train"]["distribution"]
        dist_val = stats["val"]["distribution"]
        for pid in train_ids:
            total = counts[pid]
            self.assertEqual(int(total * 0.5 / 0.7), dist_train[pid][0])
            self.assertEqual(int(total * 0.2 / 0.7), dist_val[pid][0])

        test_ids = _get_values_present(stats["test-gallery"])
        self.assertEqual(3, len(test_ids))
        self.assertEqual(test_ids, _get_values_present(stats["test-query"]))

        dist_gallery = stats["test-gallery"]["distribution"]
        dist_query = stats["test-query"]["distribution"]
        for pid in test_ids:
            total = counts[pid]
            self.assertEqual(int(total * 0.3 / 0.7), dist_gallery[pid][0])
            self.assertEqual(int(total * 0.4 / 0.7), dist_query[pid][0])

        # random seed test
        splits = [("train", 0.5), ("test", 0.5)]
        r1 = splitter.ReidentificationSplit(source, splits, query, seed=1234)
        r2 = splitter.ReidentificationSplit(source, splits, query, seed=1234)
        r3 = splitter.ReidentificationSplit(source, splits, query, seed=4321)
        self.assertEqual(
            list(r1.get_subset("train")), list(r2.get_subset("train"))
        )
        self.assertNotEqual(
            list(r1.get_subset("train")), list(r3.get_subset("train"))
        )

        # same count for checking no error in rebalance
        config = dict()
        for i in range(100):
            label = "label%03d" % i
            config[label] = {"attrs": None, "counts": 7}
        source = self._generate_dataset(config)
        splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
        actual = splitter.ReidentificationSplit(source, splits, query)

        self.assertEqual(350, len(actual.get_subset("train")))
        self.assertEqual(140, len(actual.get_subset("val")))
        self.assertEqual(90, len(actual.get_subset("test-gallery")))
        self.assertEqual(120, len(actual.get_subset("test-query")))

    def test_split_for_reidentification_with_attr(self):
        counts = {i: (i % 3 + 1) * 7 for i in range(10)}
        config = {"person": {"attrs": ["PID"], "counts": counts}}
        source = self._generate_dataset(config)

        splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
        query = 0.4 / 0.7
        actual = splitter.ReidentificationSplit(source, splits, query, "PID")

        stats = dict()
        for sname in ["train", "val", "test-query", "test-gallery"]:
            subset = actual.get_subset(sname)
            stat_subset = compute_ann_statistics(subset)["annotations"]
            stat_attr = stat_subset["labels"]["attributes"]["PID"]
            stats[sname] = stat_attr

        self.assertEqual(65, stats["train"]["count"])
        self.assertEqual(26, stats["val"]["count"])
        self.assertEqual(18, stats["test-gallery"]["count"])
        self.assertEqual(24, stats["test-query"]["count"])

        train_ids = stats["train"]["values present"]
        self.assertEqual(7, len(train_ids))
        self.assertEqual(train_ids, stats["val"]["values present"])

        trainval = stats["train"]["count"] + stats["val"]["count"]
        self.assertEqual(int(trainval * 0.5 / 0.7), stats["train"]["count"])
        self.assertEqual(int(trainval * 0.2 / 0.7), stats["val"]["count"])

        dist_train = stats["train"]["distribution"]
        dist_val = stats["val"]["distribution"]
        for pid in train_ids:
            total = counts[int(pid)]
            self.assertEqual(int(total * 0.5 / 0.7), dist_train[pid][0])
            self.assertEqual(int(total * 0.2 / 0.7), dist_val[pid][0])

        test_ids = stats["test-gallery"]["values present"]
        self.assertEqual(3, len(test_ids))
        self.assertEqual(test_ids, stats["test-query"]["values present"])

        dist_gallery = stats["test-gallery"]["distribution"]
        dist_query = stats["test-query"]["distribution"]
        for pid in test_ids:
            total = counts[int(pid)]
            self.assertEqual(int(total * 0.3 / 0.7), dist_gallery[pid][0])
            self.assertEqual(int(total * 0.4 / 0.7), dist_query[pid][0])

    def test_split_for_reidentification_gives_error(self):
        query = 0.4 / 0.7  # valid query ratio

        with self.subTest("no label"):
            source = Dataset.from_iterable([
                DatasetItem(1, annotations=[]),
                DatasetItem(2, annotations=[]),
            ], categories=["a", "b", "c"])

            with self.assertRaisesRegex(Exception, "exactly one is expected"):
                splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
                actual = splitter.ReidentificationSplit(source, splits, query)
                len(actual.get_subset("train"))

        with self.subTest(msg="multi label"):
            source = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0), Label(1)]),
                DatasetItem(2, annotations=[Label(0), Label(2)]),
            ], categories=["a", "b", "c"])

            with self.assertRaisesRegex(Exception, "exactly one is expected"):
                splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
                actual = splitter.ReidentificationSplit(source, splits, query)
                len(actual.get_subset("train"))

        counts = {i: (i % 3 + 1) * 7 for i in range(10)}
        config = {"person": {"attrs": ["PID"], "counts": counts}}
        source = self._generate_dataset(config)
        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("val", 0.2), ("test", 0.3)]
                splitter.ReidentificationSplit(source, splits, query)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.6), ("val", 0.2), ("test", 0.3)]
                splitter.ReidentificationSplit(source, splits, query)

            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
                actual = splitter.ReidentificationSplit(source, splits, -query)

        with self.subTest("wrong subset name"):
            with self.assertRaisesRegex(Exception, "Subset name"):
                splits = [("_train", 0.5), ("val", 0.2), ("test", 0.3)]
                splitter.ReidentificationSplit(source, splits, query)

        with self.subTest("wrong attribute name for person id"):
            splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
            actual = splitter.ReidentificationSplit(source, splits, query)

            with self.assertRaisesRegex(Exception, "Unknown subset"):
                actual.get_subset("test")

    def _generate_detection_dataset(self, **kwargs):
        append_bbox = kwargs.get("append_bbox")
        with_attr = kwargs.get("with_attr", False)
        nimages = kwargs.get("nimages", 10)

        label_cat = LabelCategories()
        for i in range(6):
            label = "label%d" % (i + 1)
            if with_attr is True:
                attributes = {"attr0", "attr%d" % (i + 1)}
            else:
                attributes = {}
            label_cat.add(label, attributes=attributes)
        categories = {AnnotationType.label: label_cat}

        iterable = []
        attr_val = 0
        totals = np.zeros(3)
        objects = [(1, 5, 2), (3, 4, 1), (2, 3, 4), (1, 1, 1), (2, 4, 2)]
        for img_id in range(nimages):
            cnts = objects[img_id % len(objects)]
            totals += cnts
            annotations = []
            for label_id, count in enumerate(cnts):
                attributes = {}
                if with_attr:
                    attr_val += 1
                    attributes["attr0"] = attr_val % 3
                    attributes["attr%d" % (label_id + 1)] = attr_val % 2
                for ann_id in range(count):
                    append_bbox(annotations, label_id=label_id, ann_id=ann_id,
                        attributes=attributes)
            item = DatasetItem(img_id, subset=self._get_subset(img_id),
                annotations=annotations, attributes={"id": img_id})
            iterable.append(item)

        dataset = Dataset.from_iterable(iterable, categories)
        return dataset, totals

    @staticmethod
    def _get_append_bbox(dataset_type):
        def append_bbox_coco(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )

        def append_bbox_voc(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"],
                    id=kwargs["ann_id"] + 1,
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                )
            )  # obj
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"] + 3,
                    group=kwargs["ann_id"],
                )
            )  # part
            annotations.append(
                Label(kwargs["label_id"] + 3, attributes=kwargs["attributes"])
            )

        def append_bbox_yolo(annotations, **kwargs):
            annotations.append(Bbox(1, 1, 2, 2, label=kwargs["label_id"]))
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )

        def append_bbox_cvat(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                    group=kwargs["ann_id"],
                    z_order=kwargs["ann_id"],
                )
            )
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )

        def append_bbox_labelme(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"],
                    id=kwargs["ann_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )

        def append_bbox_mot(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, label=kwargs["label_id"],
                    attributes=kwargs["attributes"],
                )
            )
            annotations.append(
                Label(kwargs["label_id"], attributes=kwargs["attributes"])
            )

        def append_bbox_widerface(annotations, **kwargs):
            annotations.append(
                Bbox(1, 1, 2, 2, attributes=kwargs["attributes"])
            )
            annotations.append(Label(0, attributes=kwargs["attributes"]))

        functions = {
            "coco": append_bbox_coco,
            "voc": append_bbox_voc,
            "yolo": append_bbox_yolo,
            "cvat": append_bbox_cvat,
            "labelme": append_bbox_labelme,
            "mot": append_bbox_mot,
            "widerface": append_bbox_widerface,
        }

        func = functions.get(dataset_type, append_bbox_cvat)
        return func

    def test_split_for_detection(self):
        dtypes = ["coco", "voc", "yolo", "cvat", "labelme", "mot", "widerface"]
        params = []
        for dtype in dtypes:
            for with_attr in [False, True]:
                params.append((dtype, with_attr, 10, 5, 3, 2))
                params.append((dtype, with_attr, 10, 7, 0, 3))

        for dtype, with_attr, nimages, train, val, test in params:
            source, _ = self._generate_detection_dataset(
                append_bbox=self._get_append_bbox(dtype),
                with_attr=with_attr,
                nimages=nimages,
            )
            total = np.sum([train, val, test])
            splits = [
                ("train", train / total),
                ("val", val / total),
                ("test", test / total),
            ]
            with self.subTest(
                dtype=dtype,
                with_attr=with_attr,
                nimage=nimages,
                train=train,
                val=val,
                test=test,
            ):
                actual = splitter.DetectionSplit(source, splits)

                self.assertEqual(train, len(actual.get_subset("train")))
                self.assertEqual(val, len(actual.get_subset("val")))
                self.assertEqual(test, len(actual.get_subset("test")))

        # random seed test
        source, _ = self._generate_detection_dataset(
            append_bbox=self._get_append_bbox("cvat"),
            with_attr=True,
            nimages=10,
        )

        splits = [("train", 0.5), ("test", 0.5)]
        r1 = splitter.DetectionSplit(source, splits, seed=1234)
        r2 = splitter.DetectionSplit(source, splits, seed=1234)
        r3 = splitter.DetectionSplit(source, splits, seed=4321)
        self.assertEqual(
            list(r1.get_subset("test")), list(r2.get_subset("test"))
        )
        self.assertNotEqual(
            list(r1.get_subset("test")), list(r3.get_subset("test"))
        )

    def test_split_for_detection_gives_error(self):
        with self.subTest(msg="bbox annotation"):
            source = Dataset.from_iterable([
                DatasetItem(1, annotations=[Label(0), Label(1)]),
                DatasetItem(2, annotations=[Label(0), Label(2)]),
            ], categories=["a", "b", "c"])

            with self.assertRaisesRegex(Exception, "more than one bbox"):
                splits = [("train", 0.5), ("val", 0.2), ("test", 0.3)]
                actual = splitter.DetectionSplit(source, splits)
                len(actual.get_subset("train"))

        source, _ = self._generate_detection_dataset(
            append_bbox=self._get_append_bbox("cvat"),
            with_attr=True,
            nimages=5,
        )

        with self.subTest("wrong ratio"):
            with self.assertRaisesRegex(Exception, "in the range"):
                splits = [("train", -0.5), ("test", 1.5)]
                splitter.DetectionSplit(source, splits)

            with self.assertRaisesRegex(Exception, "Sum of ratios"):
                splits = [("train", 0.5), ("test", 0.5), ("val", 0.5)]
                splitter.DetectionSplit(source, splits)

        with self.subTest("wrong subset name"):
            with self.assertRaisesRegex(Exception, "Subset name"):
                splits = [("train_", 0.5), ("val", 0.2), ("test", 0.3)]
                splitter.DetectionSplit(source, splits)
