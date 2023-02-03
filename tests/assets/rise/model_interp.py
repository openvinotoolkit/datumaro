from datumaro.components.annotation import Label
from datumaro.util.annotation_util import softmax


def process_outputs(inputs, outputs):
    # inputs = model input; array or images; shape = (1, H, W, C)
    # outputs = model output; shape = (1, 3);
    # results = conversion result; [[a score for label0, a score for label1, a score for label2]]

    assert len(outputs) == 1

    return [
        [
            Label(label=label, attributes={"score": score})
            for label, score in enumerate(softmax(outputs[0]))
        ]
    ]
