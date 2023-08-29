if [ -z "$1" ] ; then
    echo "No model type is given (availables: vit_h, vit_l, vit_b)"
    exit 1
fi

MODEL_TYPE=$1

echo "Build MODEL_TYPE=$1"

docker build \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \
    --build-arg model_type=${MODEL_TYPE} \
    -f Dockerfile.exporter -t segment-anything-onnx-exporter:${MODEL_TYPE} .

docker build \
    --build-context onnx-exporter=docker-image://segment-anything-onnx-exporter:${MODEL_TYPE} \
    -f Dockerfile.ovms -t segment-anything-ovms:${MODEL_TYPE} .
