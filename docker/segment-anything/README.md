# Segment Anything Model for Datumaro

## Introduction

[Segment Anything Model](https://github.com/facebookresearch/segment-anything) provides high quality object masks for the given point, bounding box, and text prompts.
[Datumaro](https://github.com/openvinotoolkit/datumaro) utilzes this zero-shot generalizable model for the dataset transformation features such as:

1. Bounding box to instance segmentation transform
2. Pseudo labeling for instance segmentation
3. ...

## Build Docker image

### Prerequisite

You should [install Docker](https://docs.docker.com/engine/install/ubuntu/) in your machine.
In addition, we recommend you to use [Ubuntu](https://ubuntu.com/) since we only provide Linux shell script for the Docker image building.

### Building Docker image for OpenVINO™ Model Server

It is not supported for now.

### Building Docker image for NVIDIA Triton™ Inference Server

1. At first, clone Datumaro repository to your local:
    ```console
    git clone https://github.com/openvinotoolkit/datumaro
    ```
2. Then, go to `docker/segment-anything` sub-directory and execute `build_triton.sh` shell script:
    ```console
    # MODEL_TYPE := "vit_h", "vit_l", or "vit_b"

    cd docker/segment-anything
    ./build_triton.sh
    ```

## Launch model server instances

We provide how to launch the dedicated model server instance for Segment Anything Model.
Because we built the Docker image for it, you can scale out it with the container orchestration system such as Kubernetes.
However, in this guide, we help you how to launch a single model server instance in your local machine.
Nevertheless, if you have multiple accelerators in your machine (e.g., 8 GPUs for a machine),
OpenVINO™ Model Server or NVIDIA Triton™ Inference Server either can use all installed devices.
For more details, please see the guide of these model server solutions.

### Building Docker image for OpenVINO™ Model Server

It is not supported for now.

### Building Docker image for NVIDIA Triton™ Inference Server

Execute the following command in your shell to launch the inference server:

```console
# MODEL_TYPE := "vit_h", "vit_l", or "vit_b"
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 segment-anything-triton-server:<MODEL_TYPE>
```
