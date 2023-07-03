# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import Dict, List, Type, Union

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from datumaro.components.abstracts.model_interpreter import ModelPred
from datumaro.components.errors import DatumaroError
from datumaro.plugins.inference_server_plugin.base import (
    LauncherForDedicatedInferenceServer,
    ProtocolType,
)

__all__ = ["TritonLauncher"]

TClient = Union[grpcclient.InferenceServerClient, httpclient.InferenceServerClient]
TInferInput = Union[grpcclient.InferInput, httpclient.InferInput]
TInferOutput = Union[grpcclient.InferResult, httpclient.InferResult]


class TritonLauncher(LauncherForDedicatedInferenceServer[TClient]):
    """Inference launcher for Triton Inference Server (https://github.com/triton-inference-server)

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
        creds = self.tls_config.as_grpc_creds() if self.tls_config is not None else None

        if self.protocol_type == ProtocolType.grpc:
            return grpcclient.InferenceServerClient(url=self.url, creds=creds)
        if self.protocol_type == ProtocolType.http:
            return httpclient.InferenceServerClient(url=self.url)

        raise NotImplementedError(self.protocol_type)

    def _check_server_health(self) -> None:
        status = self._client.is_model_ready(
            model_name=self.model_name,
            model_version=str(self.model_version),
        )
        if not status:
            raise DatumaroError("Model is not ready.")
        log.info(f"Health check succeeded: {status}")

    def _init_metadata(self) -> None:
        self._metadata = self._client.get_model_metadata(
            model_name=self.model_name,
            model_version=str(self.model_version),
        )
        log.info(f"Received metadata: {self._metadata}")

    def _get_infer_input(self, inputs: np.ndarray) -> TInferInput:
        def _fix_dynamic_batch_dim(shape):
            if shape[0] == -1:
                shape[0] = inputs.shape[0]
            return shape

        def _create(infer_input_cls: Type[TInferInput]) -> TInferInput:
            infer_inputs = [
                infer_input_cls(
                    name=inp.name,
                    shape=_fix_dynamic_batch_dim(inp.shape),
                    datatype=inp.datatype,
                )
                for inp in self._metadata.inputs
            ]
            for infer_input in infer_inputs:
                infer_input.set_data_from_numpy(inputs)
            return infer_inputs

        if self.protocol_type == ProtocolType.grpc:
            return _create(grpcclient.InferInput)
        if self.protocol_type == ProtocolType.http:
            return _create(httpclient.InferInput)

        raise NotImplementedError(self.protocol_type)

    def infer(self, inputs: np.ndarray) -> List[ModelPred]:
        infer_outputs: TInferOutput = self._client.infer(
            inputs=self._get_infer_input(inputs),
            model_name=self.model_name,
            model_version=str(self.model_version),
        )

        results: Dict[str, np.ndarray] = {
            output.name: infer_outputs.as_numpy(name=output.name)
            for output in self._metadata.outputs
        }

        outputs_group_by_item = [
            {key: output for key, output in zip(results.keys(), outputs)}
            for outputs in zip(*results.values())
        ]

        return outputs_group_by_item
