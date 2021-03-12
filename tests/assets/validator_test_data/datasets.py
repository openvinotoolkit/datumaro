# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np

from datumaro.components.dataset import Dataset, DatasetItem
from datumaro.components.extractor import Bbox, Label

dataset_main = Dataset.from_iterable([
    DatasetItem(id=1, image=np.ones((5, 5, 3)), annotations=[
        Label(1, id=0, attributes={ 'x': 1, 'y': 7, 'z': 3 }),
        Bbox(1, 2, 3, 4, id=1, label=0, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
        Bbox(2, 1, 3, 4, id=2, label=0, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
    ]),
    DatasetItem(id=2, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 2, 'y': 2, 'z': 3 }),
        Bbox(2, 3, 1, 4, id=1, label=0, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
        Bbox(2, 3, 4, 1, id=2, label=0, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ]),
    DatasetItem(id=3),
    DatasetItem(id=4, image=np.ones((2, 4, 3)), annotations=[
        Label(0, id=0, attributes={ 'y': 4, 'z': 5 }),
        Label(1, id=1, attributes={ 'x': 11, 'y': 7, 'z': 5 }),
        Bbox(1, 3, 2, 4, id=2, label=0, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
        Bbox(3, 1, 4, 2, id=3, label=0, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
    ]),
    DatasetItem(id=5, image=np.ones((2, 4, 3)), annotations=[
        Label(4, id=0, attributes={ 'a': 20 }),
        Bbox(1, 2, 3, 4, id=1, label=1, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
        Bbox(2, 1, 3, 4, id=2, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
        Bbox(2, 3, 1, 4, id=3, label=1, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
    ]),
    DatasetItem(id=6, image=np.ones((2, 4, 3)), annotations=[
        Label(1, id=0, attributes={ 'x': 11, 'y': 2, 'z': 3, 'c': 6 }),
        Bbox(2, 3, 4, 1, id=1, label=1, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
    ]),
    DatasetItem(id=7, image=np.ones((2, 4, 3)), annotations=[
        Label(1, id=0, attributes={ 'x': 1, 'y': 2, 'z': 5 }),
        Bbox(1, 2, 3, 4, id=1, label=2, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
    ]),
    DatasetItem(id=8, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 7, 'y': 9, 'z': 5 }),
        Bbox(2, 1, 3, 4, id=1, label=2, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ]),
    DatasetItem(id=9, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 7, 'y': 9, 'z': 5 }),
        Bbox(2, 3, 4, 1, id=1, label=2, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
    ]),
    DatasetItem(id=10, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 2, 'y': 2, 'z': 3 }),
        Bbox(3, 1, 2, 4, id=1, label=2, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
        Bbox(3, 1, 2, 0, id=2, label=1, attributes={
            'x': 1, 'y': 2, 'z': 3
        }),
    ]),
    DatasetItem(id=11, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 2, 'y': 2, 'z': 3 }),
        Bbox(1, 3, 4, 2, id=1, label=2, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
    ]),
    DatasetItem(id=12, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 2, 'y': 9, 'z': 4 }),
        Bbox(4, 1, 2, 3, id=1, label=2, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ]),
    DatasetItem(id=13, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 2, 'y': 6, 'z': 4 }),
        Bbox(1, 3, 2, 4, id=1, label=1, attributes={
            'x': 2, 'y': 3, 'z': 1,
        }),
        Bbox(3, 1, 4, 2, id=2, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ]),
    DatasetItem(id=14, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 8, 'y': 6, 'z': 5 }),
        Bbox(4, 1, 3, 2, id=1, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
        Bbox(1, 100000, 2, 3, id=2, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ]),
    DatasetItem(id=15, image=np.ones((2, 4, 3)), annotations=[
        Label(2, id=0, attributes={ 'x': 8, 'y': 6, 'z': 4 }),
        Bbox(3, 4, 2, 1, id=1, label=2, attributes={
            'x': 2, 'y': '3', 'z': 2,
        }),
        Bbox(1, 1, 1, 1, id=2, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
        Bbox(2, 1, 1, 1, id=3, label=1, attributes={
            'x': 3, 'y': 1, 'z': 2,
        }),
    ])
], categories=[[f'label_{i}', None, { 'x', 'y', 'z' }] \
    for i in range(4)])

dataset_w_only_one_label = Dataset.from_iterable([
    DatasetItem(id=1, image=np.ones((5, 5, 3)), annotations=[
        Label(0, attributes={ 'x': 1 }),
    ])
], categories=[[f'label_{i}', None, {'x'}] for i in range(1)])

dataset_without_label_categories = Dataset.from_iterable([
    DatasetItem(id=1, image=np.ones((5, 5, 3)), annotations=[
        Label(1, attributes={ 'x': 1 }),
    ]),
])
