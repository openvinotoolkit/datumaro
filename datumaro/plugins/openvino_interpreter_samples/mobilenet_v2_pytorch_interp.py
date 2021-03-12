# Copyright (C) 2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


from datumaro.components.extractor import Label, LabelCategories, AnnotationType
from datumaro.util.annotation_util import softmax


def process_outputs(inputs, outputs):
    # inputs = model input; array or images; shape = (B, H, W, C)
    # outputs = model output; shape = (1, 1, N, 7); N is the number of detected bounding boxes.
    # det = [image_id, label(class id), conf, x_min, y_min, x_max, y_max]
    # results = conversion result; [[ Annotation, ... ], ... ]

    results = []
    for input, output in zip(inputs, outputs):
        image_results = []
        output = softmax(output).tolist()
        label = output.index(max(output))
        image_results.append(Label(label=label, attributes={"confidences": output}))

        results.append(image_results[:])

    return results


def get_categories():
    # output categories - label map etc.

    label_categories = LabelCategories()

    with open(
        "datumaro/plugins/openvino/interpreters/imagenet.class", "r"
    ) as file:
        for line in file.readlines():
            label = line.strip()
            label_categories.add(label)

    return {AnnotationType.label: label_categories}
