import logging as log
import numpy as np
import random

from unittest import TestCase
from datumaro.components.project import Dataset
from datumaro.components.extractor import (DatasetItem, 
    Label, LabelCategories, AnnotationType)

import datumaro.plugins.splitter as splitter
from datumaro.components.operations import compute_ann_statistics

class SplitterTest(TestCase):
        
    def test_split_for_classification_multi_class_no_attr(self):
        subsets = []
        for subset, count in {"":30, "a":20, "b":10}.items():
            subsets.extend([subset for _ in range(count)])
        random.shuffle(subsets)
                     
        counts = {0:10, 1:20, 2:30}
        label_id = 0 
        iterable = []  
        for label, count in counts.items():
            for _ in range(count):
                subset = subsets[label_id]
                label_id += 1
                iterable.append(DatasetItem(
                    id=str(label_id),
                    annotations=[Label(label)],
                    subset=subset
                ))

        categories = {
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(3)
            )}            
        source_dataset = Dataset.from_iterable(iterable, categories)
        
        actual = splitter.SplitforClassification(source_dataset, splits=[
            ('train', 0.7),
            ('test', 0.3),
        ])

        self.assertEqual(42, len(actual.get_subset('train')))
        self.assertEqual(18, len(actual.get_subset('test')))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))        
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(7, dist_train["label_0"][0])
        self.assertEqual(14, dist_train["label_1"][0])
        self.assertEqual(21, dist_train["label_2"][0])
        
        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset('test'))        
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(3, dist_test["label_0"][0])
        self.assertEqual(6, dist_test["label_1"][0])
        self.assertEqual(9, dist_test["label_2"][0])

    def test_split_for_classification_single_class_single_attr(self):
        subsets = []
        for subset, count in {"":30, "a":20, "b":10}.items():
            subsets.extend([subset for _ in range(count)])
        random.shuffle(subsets)
        
        counts = {0:10, 1:20, 2:30}
        label_id = 0 
        iterable = []  
        for attr, count in counts.items():
            for _ in range(count):
                subset = subsets[label_id]
                label_id += 1
                iterable.append(DatasetItem(
                    id=str(label_id),
                    annotations=[Label(0, attributes={'attr':attr})],
                    subset=subset
                ))

        label_cat = LabelCategories()
        label_cat.add('label', attributes=['attr'])        
        categories = { AnnotationType.label: label_cat }
        
        source_dataset = Dataset.from_iterable(iterable, categories)
        
        actual = splitter.SplitforClassification(source_dataset, splits=[
            ('train', 0.7),
            ('test', 0.3),
        ])

        self.assertEqual(42, len(actual.get_subset('train')))
        self.assertEqual(18, len(actual.get_subset('test')))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))                        
        attr_train = stat_train["annotations"]["labels"]["attributes"]
        self.assertEqual(7, attr_train['attr']["distribution"]["0"][0])
        self.assertEqual(14, attr_train['attr']["distribution"]["1"][0])
        self.assertEqual(21, attr_train['attr']["distribution"]["2"][0])
        
        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset('test'))        
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(3, attr_test['attr']["distribution"]["0"][0])
        self.assertEqual(6, attr_test['attr']["distribution"]["1"][0])
        self.assertEqual(9, attr_test['attr']["distribution"]["2"][0])     
    
    def test_split_for_classification_single_class_multi_attr(self):
        subsets = []
        for subset, count in {"":60, "a":40, "b":20}.items():
            subsets.extend([subset for _ in range(count)])
        random.shuffle(subsets)
        
        counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        label_id = 0 
        iterable = []  
        for (attr1, attr2), count in counts.items():
            for _ in range(count):
                subset = subsets[label_id]
                label_id += 1
                iterable.append(DatasetItem(
                    id=str(label_id),
                    annotations=[Label(0, 
                        attributes={'attr1':attr1, 'attr2':attr2})
                    ],
                    subset=subset
                ))

        label_cat = LabelCategories()
        label_cat.add('label', attributes=['attr1', 'attr2'])        
        categories = { AnnotationType.label: label_cat }
        
        source_dataset = Dataset.from_iterable(iterable, categories)
        
        actual = splitter.SplitforClassification(source_dataset, splits=[
            ('train', 0.7),
            ('test', 0.3),
        ])

        self.assertEqual(84, len(actual.get_subset('train')))
        self.assertEqual(36, len(actual.get_subset('test')))
        
        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))       
        attr_train = stat_train["annotations"]["labels"]["attributes"] 
        self.assertEqual(49, attr_train["attr1"]["distribution"]["0"][0])
        self.assertEqual(35, attr_train["attr1"]["distribution"]["1"][0])
        self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])        
        self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])        
        self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])
                
        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset('test'))        
        attr_test = stat_test["annotations"]["labels"]["attributes"]
        self.assertEqual(21, attr_test["attr1"]["distribution"]["0"][0])
        self.assertEqual(15, attr_test["attr1"]["distribution"]["1"][0])
        self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])        
        self.assertEqual( 9, attr_test["attr2"]["distribution"]["1"][0])        
        self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])
            
    def test_split_for_classification_multi_label_with_attr(self):
        subsets = []
        for subset, count in {"":120, "a":80, "b":40}.items():
            subsets.extend([subset for _ in range(count)])
        random.shuffle(subsets)
       
        counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}        
        label_id = 0 
        iterable = []  
        for label in range(2):
            for (attr1, attr2), count in counts.items():
                if label==0:
                    attributes = {'attr1':attr1, 'attr2':attr2}
                else:
                    attributes = {'attr1':attr1, 'attr3':attr2}
                for _ in range(count):
                    subset = subsets[label_id]
                    label_id += 1
                    iterable.append(DatasetItem(
                        id=str(label_id),
                        annotations=[ Label(label, attributes=attributes) ],
                        subset=subset
                    ))
        label_cat = LabelCategories()
        label_cat.add('label0', attributes=['attr1', 'attr2'])
        label_cat.add('label1', attributes=['attr1', 'attr3'])                
        categories = { AnnotationType.label: label_cat }
        
        source_dataset = Dataset.from_iterable(iterable, categories)
    
        actual = splitter.SplitforClassification(source_dataset, splits=[
            ('train', 0.7),
            ('test', 0.3),
        ])

        self.assertEqual(168, len(actual.get_subset('train')))
        self.assertEqual(72, len(actual.get_subset('test')))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))        
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(84, dist_train["label0"][0])
        self.assertEqual(84, dist_train["label1"][0])    
        attr_train = stat_train["annotations"]["labels"]["attributes"]         
        self.assertEqual(49*2, attr_train["attr1"]["distribution"]["0"][0])
        self.assertEqual(35*2, attr_train["attr1"]["distribution"]["1"][0])
        self.assertEqual(28, attr_train["attr2"]["distribution"]["0"][0])        
        self.assertEqual(21, attr_train["attr2"]["distribution"]["1"][0])        
        self.assertEqual(35, attr_train["attr2"]["distribution"]["2"][0])
        self.assertEqual(28, attr_train["attr3"]["distribution"]["0"][0])        
        self.assertEqual(21, attr_train["attr3"]["distribution"]["1"][0])        
        self.assertEqual(35, attr_train["attr3"]["distribution"]["2"][0])
        
        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset('test'))        
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(36, dist_test["label0"][0])
        self.assertEqual(36, dist_test["label0"][0])    
        attr_test = stat_test["annotations"]["labels"]["attributes"]        
        self.assertEqual(21*2, attr_test["attr1"]["distribution"]["0"][0])
        self.assertEqual(15*2, attr_test["attr1"]["distribution"]["1"][0])
        self.assertEqual(12, attr_test["attr2"]["distribution"]["0"][0])        
        self.assertEqual( 9, attr_test["attr2"]["distribution"]["1"][0])        
        self.assertEqual(15, attr_test["attr2"]["distribution"]["2"][0])
        self.assertEqual(12, attr_test["attr3"]["distribution"]["0"][0])        
        self.assertEqual( 9, attr_test["attr3"]["distribution"]["1"][0])        
        self.assertEqual(15, attr_test["attr3"]["distribution"]["2"][0])
            

    def test_split_for_classification_gives_error_on_multi_label(self):
        iterable = [
            DatasetItem(
                id='1',
                annotations=[Label(0), Label(1)],
                subset=""
            ),
            DatasetItem(
                id='2',
                annotations=[Label(0), Label(2)],
                subset=""
            )
        ]  
        categories = {
            AnnotationType.label: LabelCategories.from_iterable(
                'label_' + str(label) for label in range(3)
            )}            
        source_dataset = Dataset.from_iterable(iterable, categories)
        with self.assertRaises(Exception):
            actual = splitter.SplitforClassification(source_dataset, splits=[
                ('train', 0.7),
                ('test', 0.3),
            ])

    def test_split_for_classification_gives_error_on_wrong_ratios(self):
        source_dataset = Dataset.from_iterable([DatasetItem(id=1)])

        with self.assertRaises(Exception):
            splitter.SplitforClassification(source_dataset, splits=[
                ('train', 0.5),
                ('test', 0.7),
            ])

        with self.assertRaises(Exception):
            splitter.SplitforClassification(source_dataset, splits=[])

        with self.assertRaises(Exception):
            splitter.SplitforClassification(source_dataset, splits=[
                ('train', -0.5),
                ('test', 1.5),
            ])

