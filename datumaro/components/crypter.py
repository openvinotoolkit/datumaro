# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
from cryptography.fernet import Fernet, InvalidToken


class Crypter:
    FERNET_KEY_LEN = 44

    def __init__(self, key: bytes) -> None:
        self._key = key
        self._fernet = Fernet(self._key)

    @property
    def key(self) -> bytes:
        return self._key

    def decrypt(self, msg: bytes) -> bytes:
        return self._fernet.decrypt(msg)

    def encrypt(self, msg: bytes) -> bytes:
        return self._fernet.encrypt(msg)

    def handshake(self, key: bytes) -> bool:
        try:
            return self.decrypt(key) == self._key
        except InvalidToken as e:
            log.debug(e)
            return False

    @staticmethod
    def gen_key() -> bytes:
        return Fernet.generate_key()

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Crypter):
            return False

        return self._key == __o._key

    @property
    def is_null_crypter(self):
        return self == NULL_CRYPTER


class NullCrypter(Crypter):
    def __init__(self) -> None:
        self._key = None
        self._fernet = None

    def decrypt(self, msg: bytes) -> bytes:
        return msg

    def encrypt(self, msg: bytes) -> bytes:
        return msg

    def handshake(self, key: bytes) -> bool:
        return self.decrypt(key) == b""


NULL_CRYPTER = NullCrypter()
