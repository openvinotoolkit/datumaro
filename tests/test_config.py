from io import StringIO
from unittest import TestCase
import os
import os.path as osp

import yaml

from datumaro.components.config import Config, DictConfig, SchemaBuilder
from datumaro.util.test_utils import TestDir

from .requirements import Requirements, mark_requirement


class ConfigTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_produce_multilayer_config_from_dict(self):
        schema_low = SchemaBuilder() \
            .add('options', dict) \
            .build()
        schema_mid = SchemaBuilder() \
            .add('desc', lambda: Config(schema=schema_low)) \
            .build()
        schema_top = SchemaBuilder() \
            .add('container', lambda: DictConfig(
                lambda v: Config(v, schema=schema_mid))) \
            .build()

        value = 1
        conf = Config({
            'container': {
                'elem': {
                    'desc': {
                        'options': {
                            'k': value
                        }
                    }
                }
            }
        }, schema=schema_top)

        self.assertEqual(value, conf.container['elem'].desc.options['k'])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        with TestDir() as test_dir:
            schema_low = SchemaBuilder() \
                .add('options', dict) \
                .build()
            schema_mid = SchemaBuilder() \
                .add('desc', lambda: Config(schema=schema_low)) \
                .build()
            schema_top = SchemaBuilder() \
                .add('container', lambda: DictConfig(
                    lambda v: Config(v, schema=schema_mid))) \
                .build()

            source = Config({
                'container': {
                    'elem': {
                        'desc': {
                            'options': {
                                'k': (1, 2, 3),
                                'd': 'asfd',
                            }
                        }
                    }
                }
            }, schema=schema_top)
            p = osp.join(test_dir, 'f.yaml')

            source.dump(p)

            loaded = Config.parse(p, schema=schema_top)

            self.assertTrue(isinstance(
                loaded.container['elem'].desc.options['k'], list))
            loaded.container['elem'].desc.options['k'] = \
                tuple(loaded.container['elem'].desc.options['k'])
            self.assertEqual(source, loaded)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_set_incorrect_key(self):
        schema = SchemaBuilder() \
            .add('k', int) \
            .build()

        with self.assertRaises(KeyError):
            Config({ 'v': 11 }, schema=schema)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_set_incorrect_value(self):
        schema = SchemaBuilder() \
            .add('k', int) \
            .build()

        with self.assertRaises(ValueError):
            Config({ 'k': 'srf' }, schema=schema)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_change_immutable(self):
        conf = Config({ 'x': 42 }, mutable=False)

        with self.assertRaises(ValueError):
            conf.y = 5

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_dump_custom_types(self):
        # The reason for this is safety.
        class X:
            pass
        conf = Config({ 'x': X() })

        with self.assertRaises(yaml.representer.RepresenterError):
            conf.dump(StringIO())

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cant_import_custom_types(self):
        # The reason for this is safety. The problem is mostly about
        # importing, because it can result in remote code execution or
        # cause unpredictable problems

        s = StringIO()
        yaml.dump({ 'x': os.system }, s, Dumper=yaml.Dumper)
        s.seek(0)

        with self.assertRaises(yaml.constructor.ConstructorError):
            Config.parse(s)
