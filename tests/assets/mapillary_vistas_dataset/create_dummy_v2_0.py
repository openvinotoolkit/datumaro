import os.path as osp
import json
import numpy as np
import os
from collections import OrderedDict

from datumaro.util.image import save_image

class Color:
    def __init__(self, r, g, b) -> None:
        self.r = r
        self.g = g
        self.b = b

label_0 = 'animal--bird'
label_1 = 'construction--barrier--separator'
label_2 = 'object--vehicle--bicycle'

colormap = OrderedDict({
    label_0: [165, 42, 42],
    label_1: [128, 128, 128],
    label_2: [119, 11, 32]
})

labelmap = OrderedDict({
    label_0: 0,
    label_1: 1,
    label_2: 2
})

categorymap = OrderedDict({
    label_0: 1,
    label_1: 10,
    label_2: 100
})


def save_json(path, data):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def make_label_mask(labels):
    return np.swapaxes(np.array([
            np.array([labelmap[label]] * 5)
            for label in labels
        ]).reshape((5, 5)), 0, 1)

def make_instance_mask(labels):
    counts = {label: 0 for label in labelmap}
    a = []
    prev_label = labels[0]
    for label in labels:
        a.append([[(labelmap[label] << 8) + counts[label]] * 5])
        if prev_label != label:
            counts[prev_label] = counts[prev_label] + 1
        prev_label = label
    return np.swapaxes(np.array(a), 0, 2).reshape((5, 5))

def make_panoptic_mask(category_ids):
    R = np.array([
        np.array([[id]] * 5)
        for id in category_ids
    ])

    G = np.zeros((5, 5, 1))
    B = np.zeros((5, 5, 1))

    res = np.swapaxes(np.array([R, G, B]).reshape((3, 5, 5)), 0, 2)
    res[:, :, :3] = res[:, :, 2::-1] # BGR to RGB
    return res

polygons0 = {
    'heigth': 5,
    'width': 5,
    'objects': [
        {
            'id': 0,
            'label': label_0,
            'polygon': [
                [0, 0, 1, 0, 2, 0, 2, 4, 0, 4]
            ]
        },
        {
            'id': 1,
            'label': label_1,
            'polygon': [
                [3, 0, 4, 0, 4, 1, 4, 4, 3, 4]
            ]
        },
    ],
    'annotation_date': ''
}

polygons1 = {
    'heigth': 5,
    'width': 5,
    'objects': [
        {
            'id': 0,
            'label': label_2,
            'polygon': [
                [0, 0, 1, 0, 1, 4, 4, 0, 0, 0]
            ]
        },
        {
            'id': 1,
            'label': label_1,
            'polygon': [
                [2, 0, 2, 1, 2, 2, 2, 3, 2, 4]
            ]
        },
        {
            'id': 2,
            'label': label_2,
            'polygon': [
                [3, 0, 4, 0, 4, 4, 3, 4, 3, 0]
            ]
        },
    ],
    'annotation_date': ''
}

polygons2 = {
    'heigth': 5,
    'width': 5,
    'objects': [
        {
            'id': 0,
            'label': label_0,
            'polygon': [
                [0, 0, 0, 1, 0, 2, 0, 3, 0, 4]
            ]
        },
        {
            'id': 1,
            'label': label_1,
            'polygon': [
                [1, 0, 1, 1, 1, 2, 1, 3, 1, 4]
            ]
        },
        {
            'id': 2,
            'label': label_0,
            'polygon': [
                [2, 0, 2, 1, 2, 2, 2, 3, 2, 4]
            ]
        },
        {
            'id': 3,
            'label': label_1,
            'polygon': [
                [3, 0, 3, 1, 3, 2, 3, 3, 3, 4]
            ]
        },
        {
            'id': 4,
            'label': label_0,
            'polygon': [
                [4, 0, 4, 1, 4, 2, 4, 3, 4, 4]
            ]
        },
    ],
    'annotation_date': ''
}

val_panoptic_config = {
    'annotations': [
        {
            'image_id': '0',
            'file_name': '0.png',
            'segments_info': [
                {
                    'id': 1,
                    'category_id': 1,
                    'area': 15,
                    'bbox': [],
                    'iscrowd': 1
                },
                {
                    'id': 2,
                    'category_id': 10,
                    'area': 10,
                    'bbox': [],
                    'iscrowd': 0
                },
             ]
        },
        {
            'image_id': '1',
            'file_name': '1.png',
            'segments_info': [
                {
                    'id': 1,
                    'category_id': 100,
                    'area': 10,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 2,
                    'category_id': 10,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 3,
                    'category_id': 100,
                    'area': 10,
                    'bbox': [],
                    'iscrowd': 1
                },
              ]
        }
    ],
    'categories': [
        {
            'id': 1,
            'name': label_0,
            'supercategory': 'animal',
            'isthing': 1,
            'color': colormap[label_0]
        },
        {
            'id': 10,
            'name': label_1,
            'supercategory': 'construction',
            'isthing': 1,
            'color': colormap[label_1]
        },
        {
            'id': 100,
            'name': label_2,
            'supercategory': 'object',
            'isthing': 1,
            'color': colormap[label_2]
        }
    ],
    'info': {
        'description': 'Dummy Mapillary Vistas dataset',
        'mapping': '',
        'version': '2.0',
        'year': '',
        'contributor': '',
        'date_created': ''
    },
    'images': [
        {
            'file_name': '0.jpg',
            'width': 5,
            'id': '0',
            'height': 5
        },
        {
            'file_name': '1.jpg',
            'width': 5,
            'id': '1',
            'height': 5
        },
    ]
}

train_panoptic_config = {
    'annotations': [
        {
            'image_id': '2',
            'file_name': '2.png',
            'segments_info': [
                {
                    'id': 1,
                    'category_id': 1,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 2,
                    'category_id': 10,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 3,
                    'category_id': 1,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 4,
                    'category_id': 10,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
                {
                    'id': 5,
                    'category_id': 1,
                    'area': 5,
                    'bbox': [],
                    'iscrowd': 0
                },
             ]
        }
    ],
    'categories': [
        {
            'id': 1,
            'name': label_0,
            'supercategory': 'animal',
            'isthing': 1,
            'color': colormap[label_0]
        },
        {
            'id': 10,
            'name': label_1,
            'supercategory': 'construction',
            'isthing': 1,
            'color': colormap[label_1]
        },
        {
            'id': 100,
            'name': label_2,
            'supercategory': 'object',
            'isthing': 1,
            'color': colormap[label_2]
        }
    ],
    'info': {
        'description': 'Dummy Mapillary Vistas dataset',
        'mapping': '',
        'version': '2.0',
        'year': '',
        'contributor': '',
        'date_created': ''
    },
    'images': [
        {
            'file_name': '2.jpg',
            'width': 5,
            'id': '2',
            'height': 5
        }
    ]
}

config = {
    'labels': [
        {
            'color': colormap[label_0],
            'instances': True,
            'readable': 'Bird',
            'name': label_0,
            'evaluate': True
        },
        {
            'color': colormap[label_1],
            'instances': True,
            'readable': 'Separator',
            'name': label_1,
            'evaluate': True
        },
        {
            'color': colormap[label_2],
            'instances': True,
            'readable': 'Bicycle',
            'name': label_2,
            'evaluate': True
        },
    ],
    'version': '1.1',
    'mapping': 'public',
    'folder_structure': ''
}

labels_pattern0 = [label_0, label_0, label_0, label_1, label_1]
labels_pattern1 = [label_2, label_2, label_1, label_2, label_2]
labels_pattern2 = [label_0, label_1, label_0, label_1, label_0]

cats_pattern0 = [categorymap[i] for i in labels_pattern0]
cats_pattern1 = [categorymap[i] for i in labels_pattern1]
cats_pattern2 = [categorymap[i] for i in labels_pattern2]

label_mask0 = make_label_mask(labels_pattern0)
label_mask1 = make_label_mask(labels_pattern1)
label_mask2 = make_label_mask(labels_pattern2)

instance_mask0 = make_instance_mask(labels_pattern0)
instance_mask1 = make_instance_mask(labels_pattern1)
instance_mask2 = make_instance_mask(labels_pattern2)

panoptic_mask0 = make_panoptic_mask([1, 1, 1, 2, 2])
panoptic_mask1 = make_panoptic_mask([1, 1, 2, 3, 3])
panoptic_mask2 = make_panoptic_mask([1, 2, 3, 4, 5])

# save_image('v2.0/val/v2.0/labels/0.png', label_mask0, create_dir=True)
# save_image('v2.0/val/v2.0/labels/1.png', label_mask1, create_dir=True)
# save_image('v2.0/train/v2.0/labels/2.png', label_mask2, create_dir=True)

# save_image('v2.0/val/v2.0/instances/0.png', instance_mask0, create_dir=True, dtype=np.int32)
# save_image('v2.0/val/v2.0/instances/1.png', instance_mask1, create_dir=True, dtype=np.int32)
# save_image('v2.0/train/v2.0/instances/2.png', instance_mask2, create_dir=True, dtype=np.int32)

# save_image('v2.0/val/v2.0/panoptic/0.png', panoptic_mask0, create_dir=True)
# save_image('v2.0/val/v2.0/panoptic/1.png', panoptic_mask1, create_dir=True)
# save_image('v2.0/train/v2.0/panoptic/2.png', panoptic_mask2, create_dir=True)

# save_image('v2.0/val/images/0.jpg', np.ones((5, 5, 3)), create_dir=True)
# save_image('v2.0/val/images/1.jpg', np.ones((5, 5, 3)), create_dir=True)
# save_image('v2.0/train/images/2.jpg', np.ones((5, 5, 3)), create_dir=True)

# save_json('v2.0/val/v2.0/polygons/0.json', polygons0)
# save_json('v2.0/val/v2.0/polygons/1.json', polygons1)
# save_json('v2.0/train/v2.0/polygons/2.json', polygons2)

# save_json('v2.0/val/v2.0/panoptic/panoptic_2020.json', val_panoptic_config)
# save_json('v2.0/train/v2.0/panoptic/panoptic_2020.json', train_panoptic_config)

save_json('v2.0/config_v2.0.json', config)