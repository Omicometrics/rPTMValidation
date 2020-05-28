import enum
import unittest

from rPTMDetermine.config import (
    Config, ConfigField, MissingConfigOptionException
)


class TestEnum(enum.Enum):
    one = enum.auto()
    two = enum.auto()


class TestConfigClass(unittest.TestCase):
    def test_config_class_constructed(self):
        class TestConfig(Config):
            _attr1_field = ConfigField('attr1')
            _attr2_field = ConfigField('attr2', True, 2)
            config_fields = [
                _attr1_field,
                _attr2_field,
                ConfigField('attr3', False, None, caster=str),
                ConfigField('attr4', True, _attr2_field),
                ConfigField('attr5', True, _attr1_field, caster=float),
                ConfigField('attr6', True, _attr1_field),
            ]

        conf = TestConfig({
            'attr1': 1,
            'attr3': 100,
            'attr6': 10
        })

        self.assertTrue(hasattr(conf, 'attr1'))
        self.assertEqual(1, conf.attr1)

        self.assertTrue(hasattr(conf, 'attr2'))
        self.assertEqual(2, conf.attr2)

        self.assertTrue(hasattr(conf, 'attr3'))
        self.assertEqual('100', conf.attr3)

        self.assertTrue(hasattr(conf, 'attr4'))
        self.assertEqual(2, conf.attr4)

        self.assertTrue(hasattr(conf, 'attr5'))
        self.assertEqual(1., conf.attr5)

        self.assertTrue(hasattr(conf, 'attr6'))
        self.assertEqual(10, conf.attr6)

    def test_config_hash_equal(self):
        """
        Tests that two equivalent Configs have the same hash.

        """
        class TestConfig(Config):
            _attr1_field = ConfigField('attr1')
            _attr2_field = ConfigField('attr2', True, 2)
            config_fields = [
                _attr1_field,
                _attr2_field,
                ConfigField('attr3', False, None, caster=str),
                ConfigField('attr4', True, _attr2_field),
                ConfigField('attr5', True, _attr1_field, caster=float)
            ]

        conf1 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })
        conf2 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })

        self.assertEqual(hash(conf1), hash(conf2))

    def test_config_hash_not_equal(self):
        """
        Tests that two differing instances of the same Config have a different
        hash.

        """
        class TestConfig(Config):
            _attr1_field = ConfigField('attr1')
            _attr2_field = ConfigField('attr2', True, 2)
            config_fields = [
                _attr1_field,
                _attr2_field,
                ConfigField('attr3', False, None, caster=str),
                ConfigField('attr4', True, _attr2_field),
                ConfigField('attr5', True, _attr1_field, caster=float)
            ]

        conf1 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })
        conf2 = TestConfig({
            'attr1': 2,
            'attr3': 100
        })

        self.assertNotEqual(hash(conf1), hash(conf2))

    def test_config_equal(self):
        """
        Tests that two equivalent Config instances test equal.

        """
        class TestConfig(Config):
            _attr1_field = ConfigField('attr1')
            _attr2_field = ConfigField('attr2', True, 2)
            config_fields = [
                _attr1_field,
                _attr2_field,
                ConfigField('attr3', False, None, caster=str),
                ConfigField('attr4', True, _attr2_field),
                ConfigField('attr5', True, _attr1_field, caster=float)
            ]

        conf1 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })
        conf2 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })

        self.assertEqual(conf1, conf2)

    def test_config_not_equal(self):
        """
        Tests that two differing instances of the same Config test not equal.

        """
        class TestConfig(Config):
            _attr1_field = ConfigField('attr1')
            _attr2_field = ConfigField('attr2', True, 2)
            config_fields = [
                _attr1_field,
                _attr2_field,
                ConfigField('attr3', False, None, caster=str),
                ConfigField('attr4', True, _attr2_field),
                ConfigField('attr5', True, _attr1_field, caster=float)
            ]

        conf1 = TestConfig({
            'attr1': 1,
            'attr3': 100
        })
        conf2 = TestConfig({
            'attr1': 2,
            'attr3': 100
        })

        self.assertNotEqual(conf1, conf2)

    def test_config_not_equal_non_instance(self):
        """
        Tests that a Config does not test equal to an object which is not
        derived from Config.

        """
        class TestConfig(Config):
            config_fields = [ConfigField('l')]

        conf = TestConfig({'l': [1]})

        self.assertNotEqual(1, conf)


    def test_config_hash_dict(self):
        """
        Tests that a config with a dictionary field can be successfully hashed.

        """
        class TestConfig(Config):
            config_fields = [ConfigField('d')]

        conf1 = TestConfig({'d': {'key': 'value'}})
        conf2 = TestConfig({'d': {'key': 'value'}})
        conf3 = TestConfig({'d': {'key': 'value2'}})

        self.assertIsInstance(hash(conf1), int)

        self.assertEqual(hash(conf1), hash(conf2))
        self.assertNotEqual(hash(conf1), hash(conf3))

    def test_config_hash_dict_field(self):
        """
        Tests that a config with a dictionary field, mapping to Config,
        can be successfully hashed

        """
        class InnerConfig(Config):
            config_fields = [ConfigField('field')]

        class TestConfig(Config):
            config_fields = [ConfigField('d')]

        conf1 = TestConfig({'d': {'key': InnerConfig({'field': 1})}})
        conf2 = TestConfig({'d': {'key': InnerConfig({'field': 1})}})
        conf3 = TestConfig({'d': {'key': InnerConfig({'field': 2})}})

        self.assertEqual(hash(conf1), hash(conf2))
        self.assertNotEqual(hash(conf1), hash(conf3))

    def test_config_hash_list(self):
        """
        Tests that a config with a list field can be successfully hashed.

        """
        class TestConfig(Config):
            config_fields = [ConfigField('l')]

        conf1 = TestConfig({'l': ['one']})
        conf2 = TestConfig({'l': ['one']})
        conf3 = TestConfig({'l': ['two']})

        self.assertIsInstance(hash(conf1), int)

        self.assertEqual(hash(conf1), hash(conf2))
        self.assertNotEqual(hash(conf1), hash(conf3))

    def test_missing_required_option(self):
        """
        Tests that missing required options are detected.

        """
        class TestConfig(Config):
            config_fields = [ConfigField('newfield')]

        with self.assertRaises(MissingConfigOptionException):
            TestConfig({})

    def test_config_string(self):
        """
        Tests string conversion for the config.

        """
        class InnerConfig(Config):
            config_fields = [ConfigField('field')]

        class TestConfig(Config):
            config_fields = [
                ConfigField('stringField'),
                ConfigField('intField'),
                ConfigField('floatField'),
                ConfigField('dictField'),
                ConfigField('dictNestedConfigField'),
                ConfigField('enumField', caster=lambda v: TestEnum[v]),
            ]

        conf = TestConfig({
            'stringField': 'teststring',
            'intField': 1,
            'floatField': 0.1,
            'dictField': {'key': 12},
            'dictNestedConfigField': {'key': InnerConfig({'field': 'value'})},
            'enumField': 'one'
        })

        expected_str = """<TestConfig stringField = teststring
intField = 1
floatField = 0.1
dictField = {'key': 12}
dictNestedConfigField = {'key': <InnerConfig field = value>}
enumField = one>"""

        self.assertEqual(expected_str, str(conf))


if __name__ == '__main__':
    unittest.main()
