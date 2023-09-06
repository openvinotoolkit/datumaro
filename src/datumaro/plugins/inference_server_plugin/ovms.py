# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import List, Union

import numpy as np
from ovmsclient import make_grpc_client, make_http_client
from ovmsclient.tfs_compat.grpc.serving_client import GrpcClient
from ovmsclient.tfs_compat.http.serving_client import HttpClient

from datumaro.components.abstracts.model_interpreter import LauncherInputType, ModelPred
from datumaro.components.errors import DatumaroError
from datumaro.plugins.inference_server_plugin.base import (
    LauncherForDedicatedInferenceServer,
    ProtocolType,
)

__all__ = ["OVMSLauncher"]

TClient = Union[GrpcClient, HttpClient]


class OVMSLauncher(LauncherForDedicatedInferenceServer[TClient]):
    """Inference launcher for OVMS (OpenVINO™ Model Server) (https://github.com/openvinotoolkit/model_server)

    Parameters:
        model_name: Name of the model. It should match with the model name loaded in the server instance.
        model_interpreter_path: Python source code path which implements a model interpreter.
            The model interpreter implement pre-processing of the model input and post-processing of the model output.
        model_version: Version of the model loaded in the server instance
        host: Host address of the server instance
        port: Port number of the server instance
        timeout: Timeout limit during communication between the client and the server instance
        tls_config: Configuration required if the server instance is in the secure mode
        protocol_type: Communication protocol type with the server instance
    """

    def _init_client(self) -> TClient:
        tls_config = self.tls_config.as_dict() if self.tls_config is not None else None

        if self.protocol_type == ProtocolType.grpc:
            return make_grpc_client(self.url, tls_config)
        if self.protocol_type == ProtocolType.http:
            return make_http_client(self.url, tls_config)

        raise NotImplementedError(self.protocol_type)

    def _check_server_health(self) -> None:
        status = self._client.get_model_status(
            model_name=self.model_name,
            model_version=self.model_version,
            timeout=self.timeout,
        )
        log.info(f"Health check succeeded: {status}")

    def _init_metadata(self):
        self._metadata = self._client.get_model_metadata(
            model_name=self.model_name,
            model_version=self.model_version,
            timeout=self.timeout,
        )
        log.info(f"Received metadata: {self._metadata}")

    def infer(self, inputs: LauncherInputType) -> List[ModelPred]:
        # Please see the following link for the input and output type of self._client.predict()
        # https://github.com/openvinotoolkit/model_server/blob/releases/2022/3/client/python/ovmsclient/lib/docs/grpc_client.md#method-predict
        # The input is Dict[str, np.ndarray].
        # The output is Dict[str, np.ndarray] (If the model has multiple outputs),
        # or np.ndarray (If the model has one single output).
        pred_inputs = {self._input_key: inputs} if isinstance(inputs, np.ndarray) else inputs
        results = self._client.predict(
            inputs=pred_inputs,
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
    def _input_key(self):
        if hasattr(self, "__input_key"):
            return self.__input_key

        metadata_inputs = self._metadata.get("inputs")

        if metadata_inputs is None:
            raise DatumaroError("Cannot get metadata of the outputs.")

        if len(metadata_inputs.keys()) > 1:
            raise DatumaroError(
                f"More than two model inputs are not allowed: {metadata_inputs.keys()}."
            )

        self.__input_key = next(iter(metadata_inputs.keys()))
        return self.__input_key

    @property
    def _output_key(self):
        if hasattr(self, "__output_key"):
            return self.__output_key

        metadata_outputs = self._metadata.get("outputs")

        if metadata_outputs is None:
            raise DatumaroError("Cannot get metadata of the outputs.")

        if len(metadata_outputs.keys()) > 1:
            raise DatumaroError(
                f"More than two model outputs are not allowed: {metadata_outputs.keys()}."
            )

        self.__output_key = next(iter(metadata_outputs.keys()))
        return self.__output_key
