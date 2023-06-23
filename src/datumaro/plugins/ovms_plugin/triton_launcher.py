# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from typing import List, Optional, Union

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from datumaro.components.abstracts.model_interpreter import ModelPred
from datumaro.components.errors import DatumaroError
from datumaro.plugins.ovms_plugin.launcher import OVMSClientType, TLSConfig

from .launcher import OVMSLauncher


class TritonLauncher(OVMSLauncher):
    def __init__(
        self,
        model_name: str,
        model_interpreter_path: str,
        host: str = "localhost",
        port: int = 9000,
        model_version: int = 0,
        timeout: float = 10,
        tls_config: Optional[TLSConfig] = None,
        ovms_client_type: OVMSClientType = OVMSClientType.grpc,
    ):
        super().__init__(
            model_name,
            model_interpreter_path,
            host,
            port,
            model_version,
            timeout,
            tls_config,
            ovms_client_type,
        )

    def _init_client(
        self,
        ovms_client_type,
    ) -> Union[grpcclient.InferenceServerClient, httpclient.InferenceServerClient]:
        creds = self.tls_config.as_grpc_creds() if self.tls_config is not None else None

        if ovms_client_type == OVMSClientType.grpc:
            return grpcclient.InferenceServerClient(url=self.url, creds=creds)
        elif ovms_client_type == OVMSClientType.http:
            return httpclient.InferenceServerClient(url=self.url)
        else:
            raise NotImplementedError(ovms_client_type)

    def _check_server_health(self, model_version, timeout):
        try:
            status = self._client.is_model_ready(
                model_name=self.model_name,
                model_version=str(model_version),
            )
            log.info(f"Health check succeeded: {status}")
        except Exception as e:
            raise DatumaroError(
                f"Health check failed for model_name={self.model_name}, "
                f"model_version={model_version}, url={self.url} and tls_config={self.tls_config}"
            ) from e

    def _init_input_name(self, model_version, timeout):
        # self._client: grpcclient.InferenceServerClient
        metadata = self._client.get_model_metadata(
            model_name=self.model_name,
            model_version=str(model_version),
        )
        metadata_inputs = metadata.inputs
        if metadata_inputs is None:
            raise DatumaroError("Cannot get metadata of the inputs.")

        if len(metadata_inputs) > 1:
            raise DatumaroError(f"More than two model inputs are not allowed: {metadata_inputs}.")

        self._input_key = next(iter(metadata_inputs)).name
        log.info(f"Model input key is {self._input_key}")

    def infer(self, inputs: np.ndarray) -> List[ModelPred]:
        results = self._client.predict(
            inputs={self._input_key: inputs},
            model_name=self.model_name,
            model_version=self.model_version,
        )
        client = grpcclient.InferenceServerClient(url=self.url)
        client.Infer
        grpcclient.InferInput()

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
