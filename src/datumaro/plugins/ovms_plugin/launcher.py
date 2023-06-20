# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional

import grpc
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.errors import MediaTypeError
from datumaro.components.launcher import Launcher
from datumaro.components.media import Image


class _OVMSImporter(CliPlugin):
    @staticmethod
    def _parse_output_layers(s):
        return [s.strip() for s in s.split(",")]

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "-d", "--description", required=True, help="Path to the model description file (.xml)"
        )
        parser.add_argument(
            "-w", "--weights", required=True, help="Path to the model weights file (.bin)"
        )
        parser.add_argument(
            "-i",
            "--interpreter",
            required=True,
            help="Path to the network output interprter script (.py)",
        )
        parser.add_argument("--device", default="CPU", help="Target device (default: %(default)s)")
        parser.add_argument(
            "--output-layers",
            type=cls._parse_output_layers,
            help="A comma-separated list of extra output layers",
        )
        return parser

    @staticmethod
    def copy_model(model_dir, model):
        shutil.copy(model["description"], osp.join(model_dir, osp.basename(model["description"])))
        model["description"] = osp.basename(model["description"])

        shutil.copy(model["weights"], osp.join(model_dir, osp.basename(model["weights"])))
        model["weights"] = osp.basename(model["weights"])

        shutil.copy(model["interpreter"], osp.join(model_dir, osp.basename(model["interpreter"])))
        model["interpreter"] = osp.basename(model["interpreter"])


class OVMSLauncher(Launcher):
    cli_plugin = _OVMSImporter
    ALLOWED_CHANNEL_FORMATS = {"NCHW", "NHWC"}

    def __init__(
        self,
        model_name: str,
        host: str = "localhost",
        port: int = 9000,
        credential: Optional[grpc.ChannelCredentials] = None,
        channel_format: str = "NCHW",
        to_rgb: bool = True,
    ):
        address = f"{host}:{port}"

        # server_ca_cert, client_key, client_cert = prepare_certs(
        #     server_cert=args["server_cert"],
        #     client_key=args["client_key"],
        #     client_ca=args["client_cert"],
        # )
        # creds = grpc.ssl_channel_credentials(
        #     root_certificates=server_ca_cert,
        #     private_key=client_key,
        #     certificate_chain=client_cert,
        # )

        channel = (
            grpc.secure_channel(address, credential=credential)
            if credential
            else grpc.insecure_channel(address)
        )

        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.stub.Predict.future

    def infer(self, inputs):
        inputs = self.process_inputs(inputs)
        results = self._request.infer(inputs)
        if len(results) == 1:
            return next(iter(results.values()))
        else:
            return results

    def launch(self, inputs):
        batch_size = len(inputs)
        if self._batch_size < batch_size:
            self._load_executable_net(batch_size)

        outputs = self.infer(inputs)
        results = self.process_outputs(inputs, outputs)
        return results

    def infos(self):
        return None

    def categories(self):
        return None

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
