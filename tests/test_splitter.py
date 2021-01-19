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
    def _generate_dataset(self, config):
        # counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        # attr1 = ['attr1', 'attr2']
        # attr2 = ['attr1', 'attr3']
        # config = { "label1": { "attrs": attr1, "counts": counts }, 
        #            "label2": { "attrs": attr2, "counts": counts }}        
        def _get_subset():                
            return np.random.choice(['', 'a', 'b'], p = [0.5, 0.3, 0.2])

        iterable = []  
        label_cat = LabelCategories()
        idx = 0 
        for label_id, label in enumerate(config.keys()): 
            anames = config[label]['attrs']
            counts = config[label]['counts']            
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
                        iterable.append(DatasetItem(
                            id=str(idx),
                            annotations=[ Label(label_id, attributes=attributes) ],
                            subset=_get_subset()
                        ))
            else:
                for _ in range(counts):
                    idx += 1
                    iterable.append(DatasetItem(
                        id=str(idx),
                        annotations=[Label(label_id)],
                        subset=_get_subset()
                    ))
        categories = { AnnotationType.label: label_cat }
        dataset = Dataset.from_iterable(iterable, categories)
        return dataset

    def test_split_for_classification_multi_class_no_attr(self):
        config = { "label1": { "attrs": None, "counts": 10 }, 
                   "label2": { "attrs": None, "counts": 20 },
                   "label3": { "attrs": None, "counts": 30 }}
        source_dataset = self._generate_dataset(config)
        
        actual = splitter.SplitforClassification(source_dataset, 
            train=0.7, test=0.3)            
        self.assertEqual(42, len(actual.get_subset('train')))
        self.assertEqual(18, len(actual.get_subset('test')))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))        
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(7, dist_train["label1"][0])
        self.assertEqual(14, dist_train["label2"][0])
        self.assertEqual(21, dist_train["label3"][0])
        
        # check stats for test
        stat_test = compute_ann_statistics(actual.get_subset('test'))        
        dist_test = stat_test["annotations"]["labels"]["distribution"]
        self.assertEqual(3, dist_test["label1"][0])
        self.assertEqual(6, dist_test["label2"][0])
        self.assertEqual(9, dist_test["label3"][0])

    def test_split_for_classification_single_class_single_attr(self):      
        counts = {0:10, 1:20, 2:30}        
        config = { "label": { "attrs": ['attr'], "counts": counts }}
        source_dataset = self._generate_dataset(config) 
       
        actual = splitter.SplitforClassification(source_dataset, \
            train=0.7, test=0.3)

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
        counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        attrs = ['attr1', 'attr2']
        config = { "label": { "attrs": attrs, "counts": counts }}
        source_dataset = self._generate_dataset(config) 
        
        actual = splitter.SplitforClassification(source_dataset,
            train=0.7, test=0.3)

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
        counts = {(0,0):20, (0,1):20, (0,2):30, (1,0):20, (1,1):10, (1,2):20}
        attr1 = ['attr1', 'attr2']
        attr2 = ['attr1', 'attr3']
        config = { "label1": { "attrs": attr1, "counts": counts }, 
                   "label2": { "attrs": attr2, "counts": counts }}
        source_dataset = self._generate_dataset(config) 

        actual = splitter.SplitforClassification(source_dataset,
            train=0.7, test=0.3)

        self.assertEqual(168, len(actual.get_subset('train')))
        self.assertEqual(72, len(actual.get_subset('test')))

        # check stats for train
        stat_train = compute_ann_statistics(actual.get_subset('train'))        
        dist_train = stat_train["annotations"]["labels"]["distribution"]
        self.assertEqual(84, dist_train["label1"][0])
        self.assertEqual(84, dist_train["label2"][0])    
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
        self.assertEqual(36, dist_test["label1"][0])
        self.assertEqual(36, dist_test["label2"][0])    
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
            actual = splitter.SplitforClassification(source_dataset, 
                train=0.7, test=0.3)

    def test_split_for_classification_gives_error_on_wrong_ratios(self):
        source_dataset = Dataset.from_iterable([DatasetItem(id=1)])
        with self.assertRaises(Exception):
            splitter.SplitforClassification(source_dataset, 
                train=-0.5, test=1.5)
    
    def test_split_for_matching_reid(self):
        counts = { i:(i%3+1)*7 for i in range(10) }
        config = { "person": { "attrs": ['PID'], "counts": counts }}
        source_dataset = self._generate_dataset(config)
                
        actual = splitter.SplitforMatchingReID(source_dataset, \
            train=0.5, val=0.2, test=0.3, query=0.4, gallery=0.3)
        
        stats = dict()
        for sname in ['train', 'val', 'test']:
            subset = actual.get_subset(sname)
            stat_subset = compute_ann_statistics(subset)["annotations"]
            stat_attr = stat_subset["labels"]["attributes"]["PID"]
            stats[sname] = stat_attr
        
        for sname in ['gallery', 'query']:
            subset = actual.get_subset_by_group(sname)
            stat_subset = compute_ann_statistics(subset)["annotations"]
            stat_attr = stat_subset["labels"]["attributes"]["PID"]
            stats[sname] = stat_attr
        
        self.assertEqual(65, stats['train']['count'])   # depends on heuristic
        self.assertEqual(26, stats['val']['count'])     # depends on heuristic
        self.assertEqual(42, stats['test']['count'])    # depends on heuristic
        
        train_ids = stats['train']['values present']
        self.assertEqual(7, len(train_ids))
        self.assertEqual(train_ids, stats['val']['values present'])
        
        trainval = stats['train']['count'] + stats['val']['count']
        self.assertEqual(int(trainval * 0.5 / 0.7), stats['train']['count'])
        self.assertEqual(int(trainval * 0.2 / 0.7), stats['val']['count'])

        dist_train = stats['train']['distribution']
        dist_val = stats['val']['distribution']
        for pid in train_ids:
            total = counts[int(pid)]
            self.assertEqual(int(total * 0.5 / 0.7), dist_train[pid][0])
            self.assertEqual(int(total * 0.2 / 0.7), dist_val[pid][0])
            
        test_ids = stats['test']['values present']
        self.assertEqual(3, len(test_ids))
        self.assertEqual(test_ids, stats['gallery']['values present'])
        self.assertEqual(test_ids, stats['query']['values present'])
        
        dist_test = stats['test']['distribution']
        dist_gallery = stats['gallery']['distribution']
        dist_query = stats['query']['distribution']
        for pid in test_ids:
            total = counts[int(pid)]
            self.assertEqual(total, dist_test[pid][0])
            self.assertEqual(int(total * 0.3 / 0.7), dist_gallery[pid][0])
            self.assertEqual(int(total * 0.4 / 0.7), dist_query[pid][0])
 