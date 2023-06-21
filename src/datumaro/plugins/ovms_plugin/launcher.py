# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Union

import numpy as np
from ovmsclient import make_grpc_client, make_http_client
from ovmsclient.tfs_compat.grpc.serving_client import GrpcClient
from ovmsclient.tfs_compat.http.serving_client import HttpClient

from datumaro.components.abstracts.model_interpreter import ModelPred
from datumaro.components.errors import DatumaroError, MediaTypeError
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.components.media import Image


class OVMSClientType(IntEnum):
    """API types of OVMS client

    OVMS client can accept gRPC or HTTP REST API.
    """

    grpc = 0
    http = 1


@dataclass(frozen=True)
class TLSConfig:
    """TLS configuration dataclass

    Parameters:
        client_key_path: Path to client key file
        client_cert_path: Path to client certificate file
        server_cert_path: Path to server certificate file
    """

    client_key_path: str
    client_cert_path: str
    server_cert_path: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "client_key_path": self.client_key_path,
            "client_cert_path": self.client_cert_path,
            "server_cert_path": self.server_cert_path,
        }


class OVMSLauncher(LauncherWithModelInterpreter):
    """Inference launcher for OVMS (OpenVINO™ Model Server)

    Parameters:
        model_name: Name of the model. It should match with the model name loaded in the OVMS instance.
        model_interpreter_path: Python source code path which implements a model interpreter.
            The model interpreter implement pre-processing of the model input and post-processing of the model output.
        host: Host address of the OVMS instance
        port: Port number of the OVMS instance
        model_version: Version of the model loaded in the OVMS instance
        timeout: Timeout limit during communication between the client and the OVMS instance
        tls_config: Configuration required if the OVMS instance is in the secure mode
        ovms_client_type: OVMS client API type
    """

    def __init__(
        self,
        model_name: str,
        model_interpreter_path: str,
        host: str = "localhost",
        port: int = 9000,
        model_version: int = 0,
        timeout: float = 10.0,
        tls_config: Optional[TLSConfig] = None,
        ovms_client_type: OVMSClientType = OVMSClientType.grpc,
    ):
        super().__init__(model_interpreter_path=model_interpreter_path)

        self._client = self._init_client(
            model_name,
            host,
            port,
            tls_config,
            ovms_client_type,
        )
        self._check_server_health(model_version, timeout)
        self._init_input_name(model_version, timeout)

        self.model_version = model_version
        self.timeout = timeout

    def _init_client(
        self,
        model_name,
        host,
        port,
        tls_config,
        ovms_client_type,
    ) -> Union[GrpcClient, HttpClient]:
        self.model_name = model_name
        self.url = f"{host}:{port}"
        self.tls_config = tls_config

        if ovms_client_type == OVMSClientType.grpc:
            return make_grpc_client(self.url, self.tls_config)
        elif ovms_client_type == OVMSClientType.http:
            return make_http_client(self.url, self.tls_config)
        else:
            raise NotImplementedError(ovms_client_type)

    def _check_server_health(self, model_version, timeout):
        try:
            status = self._client.get_model_status(
                model_name=self.model_name,
                model_version=model_version,
                timeout=timeout,
            )
            log.info(f"Health check succeeded: {status}")
        except Exception as e:
            raise DatumaroError(
                f"Health check failed for model_name={self.model_name}, "
                f"model_version={model_version}, url={self.url} and tls_config={self.tls_config}"
            ) from e

    def _init_input_name(self, model_version, timeout):
        metadata = self._client.get_model_metadata(
            model_name=self.model_name,
            model_version=model_version,
            timeout=timeout,
        )
        metadata_inputs = metadata.get("inputs")
        if metadata_inputs is None:
            raise DatumaroError("Cannot get metadata of the inputs.")

        if len(metadata_inputs.keys()) > 1:
            raise DatumaroError(
                f"More than two model inputs are not allowed: {metadata_inputs.keys()}."
            )

        self._input_key = next(iter(metadata_inputs.keys()))
        log.info(f"Model input key is {self._input_key}")

    def infer(self, inputs: np.ndarray) -> List[ModelPred]:
        results = self._client.predict(
            inputs={self._input_key: inputs},
            model_name=self.model_name,
            model_version=self.model_version,
            timeout=self.timeout,
        )

        # If there is only one output key,
        # it returns `np.ndarray`` rather than `Dict[str, np.ndarray]`.
        # Please see ovmsclient.tfs_compat.grpc.responses.GrpcPredictResponse
        if isinstance(results, np.ndarray):
            results = {self._output_key: results}

        outputs_group_by_item = [
            {key: output for key, output in zip(results.keys(), outputs)}
            for outputs in zip(*results.values())
        ]

        return outputs_group_by_item

    @property
    def _output_key(self):
        if not hasattr(self, "__output_key"):
            metadata = self._client.get_model_metadata(
                model_name=self.model_name,
                model_version=self.model_version,
                timeout=self.timeout,
            )
            metadata_outputs = metadata.get("outputs")

            if metadata_outputs is None:
                raise DatumaroError("Cannot get metadata of the outputs.")

            if len(metadata_outputs.keys()) > 1:
                raise DatumaroError(
                    f"More than two model outputs are not allowed: {metadata_outputs.keys()}."
                )

            self.__output_key = next(iter(metadata_outputs.keys()))

        return self.__output_key

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
