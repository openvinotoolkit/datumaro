# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Optional

from cryptography.fernet import Fernet


class Crypter:
    FERNET_KEY_LEN = 44

    def __init__(self, key: Optional[bytes]) -> None:
        if key is not None:
            self._key = key
            self._fernet = Fernet(self._key)
        else:
            self._key = None
            self._fernet = None

    @property
    def key(self) -> Optional[bytes]:
        return self._key

    def decrypt(self, msg: bytes):
        return self._fernet.decrypt(msg) if self._fernet is not None else msg

    def encrypt(self, msg: bytes):
        return self._fernet.encrypt(msg) if self._fernet is not None else msg

    def handshake(self, key: bytes) -> bool:
        if self._key is None and key == b"":
            return True
        if self._key is not None and self._key == key:
            return True

        return False

    @staticmethod
    def gen_key() -> bytes:
        return Fernet.generate_key()
