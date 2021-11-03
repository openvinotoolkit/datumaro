import numpy as np
from collections import OrderedDict

from datumaro.util.image import save_image

class Color:
    def __init__(self, r, g, b) -> None:
        self.r = r
        self.g = g
        self.b = b

colormap = OrderedDict({
    'label_0': Color(10, 50, 90),
    'label_1': Color(20, 30, 80),
    'label_2': Color(30, 70, 40)
})

labelmap = OrderedDict({
    'label_0': 0,
    'label_1': 1,
    'label_2': 2
})

def make_label_mask(labels):
    return np.swapaxes(np.array([
            np.array([labelmap[label]] * 5)
            for label in labels
        ]).reshape((5, 5)), 0, 1)
#
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


label_mask0 = make_label_mask(['label_0', 'label_0', 'label_1', 'label_1', 'label_2'])
label_mask1 = make_label_mask(['label_0', 'label_0', 'label_1', 'label_1', 'label_0'])
label_mask2 = make_label_mask(['label_1', 'label_1', 'label_2', 'label_1', 'label_1'])

instance_mask0 = make_instance_mask(['label_0', 'label_0', 'label_1', 'label_1', 'label_2'])
instance_mask1 = make_instance_mask(['label_0', 'label_0', 'label_1', 'label_1', 'label_0'])
instance_mask2 = make_instance_mask(['label_1', 'label_1', 'label_2', 'label_1', 'label_1'])

save_image('val/v1.2/labels/0.png', label_mask0, create_dir=True)
save_image('val/v1.2/instances/0.png', instance_mask0, dtype=np.int32, create_dir=True)

save_image('train/v1.2/labels/1.png', label_mask1, create_dir=True)
save_image('train/v1.2/labels/2.png', label_mask2, create_dir=True)
save_image('train/v1.2/instances/1.png', instance_mask1, dtype=np.int32, create_dir=True)