from collections import defaultdict
from inspect import isclass
from typing import Dict, Generator, Generic, Iterable, Iterator, Optional, Tuple, Type, TypeVar

from datumaro.components.cli_plugin import CliPlugin
from datumaro.components.dataset_base import DatasetBase, SubsetBase
from datumaro.components.exporter import Exporter
from datumaro.components.generator import DatasetGenerator
from datumaro.components.importer import Importer
from datumaro.components.launcher import Launcher
from datumaro.components.lazy_plugin import LazyPlugin
from datumaro.components.transformer import ItemTransform, Transform
from datumaro.components.validator import Validator

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self._items: Dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        self._items[name] = value
        return value

    def unregister(self, name: str) -> Optional[T]:
        return self._items.pop(name, None)

    def get(self, key: str) -> T:
        """Returns a class or a factory function"""
        return self._items[key]

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __contains__(self, key) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def items(self) -> Generator[Tuple[str, T], None, None]:
        for key in self:
            yield key, self.get(key)


class PluginRegistry(Registry[Type[CliPlugin]]):
    _ACCEPT: Type[CliPlugin] = None
    _SKIP: Optional[Iterable[Type[CliPlugin]]] = None
    _DECLINE: Optional[Type[CliPlugin]] = None

    def __init__(self):
        super().__init__()
        if self._ACCEPT is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires an _ACCEPT class attribute"
                " to specify the accepted type of stored instances."
            )

    def _filter(self, t):
        skip = {self.skip} if isclass(self.skip) else set(self.skip or [])
        skip = tuple(skip | set((self.accept,)))
        if (
            not issubclass(t, self.accept)
            or t in skip
            or (self.decline and issubclass(t, self.decline))
        ):
            return False
        if getattr(t, "__not_plugin__", None):
            return False
        return True

    def get(self, key: str) -> Type[CliPlugin]:
        """Returns a class or a factory function"""
        item = self._items[key]
        if issubclass(item, LazyPlugin):
            return item.get_plugin_cls()
        return item

    def batch_register(self, values: Iterable[Type[CliPlugin]]):
        for v in values:
            if not self._filter(v):
                continue

            self.register(v.NAME, v)


class DatasetBaseRegistry(PluginRegistry):
    _ACCEPT = DatasetBase
    _SKIP = (SubsetBase, Transform, ItemTransform)
    _DECLINE = Transform


class ImporterRegistry(PluginRegistry):
    _ACCEPT = Importer

    def __init__(self):
        super().__init__()
        self.extension_groups = defaultdict(list)

    def register(self, name: str, value: Type[Importer]) -> Type[Importer]:
        super().register(name, value)
        importer = self.get(name)
        for extension in importer.get_file_extensions():
            self.extension_groups[extension].append((name, importer))
        return value


class LauncherRegistry(PluginRegistry):
    _ACCEPT = Launcher


class ExporterRegistry(PluginRegistry):
    _ACCEPT = Exporter


class GeneratorRegistry(PluginRegistry):
    _ACCEPT = DatasetGenerator


class TransformRegistry(PluginRegistry):
    _ACCEPT = Transform
    _SKIP = ItemTransform


class ValidatorRegistry(PluginRegistry):
    _ACCEPT = Validator