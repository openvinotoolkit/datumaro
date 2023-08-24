# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse

import numpy as np
import torch
from segment_anything import sam_model_registry
from torch import nn

parser = argparse.ArgumentParser(description="Export the SAM image encoder to an ONNX model.")

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--output", type=str, required=True, help="The filename to save the ONNX model to."
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)


if __name__ == "__main__":
    args = parser.parse_args()

    class Encoder(nn.Module):
        def __init__(self, model_type: str, checkpoint: str) -> None:
            super().__init__()
            self.sam = sam_model_registry[model_type](checkpoint=checkpoint)

        def forward(self, img: torch.Tensor):
            x = self.sam.preprocess(img)
            return self.sam.image_encoder(x)

    inputs = {
        "img": torch.from_numpy(
            np.random.randint(0, 256, size=[1, 3, 1024, 1024], dtype=np.uint8)
        ).to(dtype=torch.float32)
    }

    output_names = ["image_embeddings"]

    with open(args.output, "wb") as f:
        torch.onnx.export(
            Encoder(model_type=args.model_type, checkpoint=args.checkpoint),
            inputs,
            f,
            export_params=True,
            verbose=False,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=list(inputs.keys()),
            output_names=output_names,
            dynamic_axes=None,
        )
