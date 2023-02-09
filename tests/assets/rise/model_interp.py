from datumaro.components.annotation import Label
from datumaro.util.annotation_util import softmax


def process_outputs(inputs, outputs):
    # inputs = model input; array or images; shape = (B, H, W, C)
    # outputs = model output; shape = (B, 3);
    # results = conversion result;
    # [B x [a score for label0, a score for label1, a score for label2]];

    return [
        [
            Label(label=label, attributes={"score": score})
            for label, score in enumerate(softmax(output))
        ]
        for output in outputs
    ]
