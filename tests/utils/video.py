# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT


import cv2
import numpy as np

from datumaro.util.scope import on_exit_do, scoped


@scoped
def make_sample_video(path, frames=4, frame_size=(10, 20), fps=25.0):
    """
    frame_size is (H, W), only even sides
    """

    writer = cv2.VideoWriter(
        path,
        frameSize=tuple(frame_size[::-1]),
        fps=float(fps),
        fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
    )
    on_exit_do(writer.release)

    for i in range(frames):
        # Apparently, only uint8 values are supported, but not floats
        # Colors are compressed, but grayscale colors suffer no loss
        writer.write(np.ones((*frame_size, 3), dtype=np.uint8) * i)
