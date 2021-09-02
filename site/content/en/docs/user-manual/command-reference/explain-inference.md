---
title: 'Run inference explanation'
linkTitle: 'Explain inference'
description: ''
weight: 22
---

Runs an explainable AI algorithm for a model.

This tool is supposed to help an AI developer to debug a model and a dataset.
Basically, it executes inference and tries to find problems in the trained
model - determine decision boundaries and belief intervals for the classifier.

Currently, the only available algorithm is RISE ([article](https://arxiv.org/pdf/1806.07421.pdf)),
which runs inference and then re-runs a model multiple times on each
image to produce a heatmap of activations for each output of the
first inference. As a result, we obtain few heatmaps, which
shows, how image pixels affected the inference result. This algorithm doesn't
require any special information about the model, but it requires the model to
return all the outputs and confidences. The algorithm only supports
classification and detection models.

The following use cases available:
- RISE for classification
- RISE for object detection

Usage:

``` bash
datum explain --help

datum explain \
    -m <model_name> \
    -o <save_dir> \
    -t <target> \
    <method> \
    <method_params>
```

Example: run inference explanation on a single image with visualization

``` bash
datum create <...>
datum model add mymodel <...>
datum explain -t image.png -m mymodel \
    rise --max-samples 1000 --progressive
```

> Note: this algorithm requires the model to return
> _all_ (or a _reasonable_ amount) the outputs and confidences unfiltered,
> i.e. all the `Label` annotations for classification models and
> all the `Bbox`es for detection models.
> You can find examples of the expected model outputs in [`tests/test_RISE.py`](https://github.com/openvinotoolkit/datumaro/blob/develop/tests/test_RISE.py)

For OpenVINO models the output processing script would look like this:

Classification scenario:

``` python
from datumaro.components.annotation import Label
from datumaro.util.annotation_util import softmax

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, logits, shape = (N, n_classes)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        confs = softmax(output[0])
        for label, conf in enumerate(confs):
            results.append(Label(int(label)), attributes={'score': float(conf)})

    return results
```


Object Detection scenario:

``` python
from datumaro.components.annotation import Bbox

# return a significant number of output boxes to make multiple runs
# statistically correct and meaningful
max_det = 1000

def process_outputs(inputs, outputs):
    # inputs = model input, array or images, shape = (N, C, H, W)
    # outputs = model output, shape = (N, 1, K, 7)
    # results = conversion result, [ [ Annotation, ... ], ... ]
    results = []
    for input, output in zip(inputs, outputs):
        input_height, input_width = input.shape[:2]
        detections = output[0]
        image_results = []
        for i, det in enumerate(detections):
            label = int(det[1])
            conf = float(det[2])
            x = max(int(det[3] * input_width), 0)
            y = max(int(det[4] * input_height), 0)
            w = min(int(det[5] * input_width - x), input_width)
            h = min(int(det[6] * input_height - y), input_height)
            image_results.append(Bbox(x, y, w, h,
                label=label, attributes={'score': conf} ))

            results.append(image_results[:max_det])

    return results
```
