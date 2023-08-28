# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Generic, Optional, TypeVar

from grpc import ChannelCredentials, ssl_channel_credentials
from ovmsclient.tfs_compat.base.serving_client import ServingClient

from datumaro.components.errors import DatumaroError, MediaTypeError
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.components.media import Image


class InferenceServerType(IntEnum):
    """Types of the dedicated inference server"""

    ovms = 0
    triton = 1


class ProtocolType(IntEnum):
    """Protocol type for communication with dedicated inference server"""

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

    def as_grpc_creds(self) -> ChannelCredentials:
        server_cert, client_cert, client_key = ServingClient._prepare_certs(
            self.server_cert_path, self.client_cert_path, self.client_key_path
        )
        return ssl_channel_credentials(
            root_certificates=server_cert, private_key=client_key, certificate_chain=client_cert
        )


TClient = TypeVar("TClient")


class LauncherForDedicatedInferenceServer(Generic[TClient], LauncherWithModelInterpreter):
    """Inference launcher for dedicated inference server

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

    def __init__(
        self,
        model_name: str,
        model_interpreter_path: str,
        model_version: int = 0,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = 10.0,
        tls_config: Optional[TLSConfig] = None,
        protocol_type: ProtocolType = ProtocolType.grpc,
    ):
        super().__init__(model_interpreter_path=model_interpreter_path)

        self.model_name = model_name
        self.model_version = model_version
        self.url = f"{host}:{port}"
        self.timeout = timeout
        self.tls_config = tls_config
        self.protocol_type = protocol_type

        try:
            self._client = self._init_client()
            self._check_server_health()
            self._init_metadata()
        except Exception as e:
            raise DatumaroError(
                f"Health check failed for model_name={self.model_name}, "
                f"model_version={self.model_version}, url={self.url} and tls_config={self.tls_config}"
            ) from e

    def _init_client(self) -> TClient:
        raise NotImplementedError()

    def _check_server_health(self) -> None:
        raise NotImplementedError()

    def _init_metadata(self) -> None:
        raise NotImplementedError()

    def type_check(self, item):
        if not isinstance(item.media, Image):
            raise MediaTypeError(f"Media type should be Image, Current type={type(item.media)}")
        return True
