# OpenVINO Inference Interpreter 
Interpreter samples to parse OpenVINO inference outputs.

## Models supported from interpreter samples

- Intel Pre-trained Models
  - Object Detection
    - face-detection-0200 (https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0200_description_face_detection_0200.html)
    - face-detection-0202 (https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0202_description_face_detection_0202.html)
    - face-detection-0204 (https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_0204_description_face_detection_0204.html)
    - person-detection-0200 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0200_description_person_detection_0200.html)
    - person-detection-0201 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0201_description_person_detection_0201.html)
    - person-detection-0202 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_0202_description_person_detection_0202.html) 
    - person-vehicle-bike-detection-2000 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2000_description_person_vehicle_bike_detection_2000.html)
    - person-vehicle-bike-detection-2001 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2001_description_person_vehicle_bike_detection_2001.html)
    - person-vehicle-bike-detection-2002 (https://docs.openvinotoolkit.org/latest/omz_models_intel_person_vehicle_bike_detection_2002_description_person_vehicle_bike_detection_2002.html)
    - vehicle-detection-0200 (https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0200_description_vehicle_detection_0200.html)
    - vehicle-detection-0201 (https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0201_description_vehicle_detection_0201.html)
    - vehicle-detection-0202 (https://docs.openvinotoolkit.org/latest/omz_models_intel_vehicle_detection_0202_description_vehicle_detection_0202.html)

- Public Pre-Trained Models(OMZ)
  - Classification
    - mobilenet-v2-pytorch (https://docs.openvinotoolkit.org/latest/omz_models_public_mobilenet_v2_pytorch_mobilenet_v2_pytorch.html)
  - Object Detection
    - ssd_mobilenet_v1_coco (https://docs.openvinotoolkit.org/latest/omz_models_public_ssd_mobilenet_v1_coco_ssd_mobilenet_v1_coco.html)
    - ssd_mobilenet_v2_coco (https://docs.openvinotoolkit.org/latest/omz_models_public_ssd_mobilenet_v2_coco_ssd_mobilenet_v2_coco.html)

## Model download
- Prerequisites: OpenVINO (For installation, please see https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html)
- To download OpenVINO models, please see https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html

  ```bash
  # cd <openvino_directory>/deployment_tools/open_model_zoo/tools/downloader
  # ./downloader.py --name <model_name>
  #
  # Examples
  cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
  ./downloader.py --name face-detection-0200
  ```