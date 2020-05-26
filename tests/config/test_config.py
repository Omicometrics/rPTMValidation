import unittest

from rPTMDetermine.config import Config, ConfigField


class TestConfigConstruction(unittest.TestCase):
    def test_config_class_constructed(self):
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

        conf = TestConfig({
            'attr1': 1,
            'attr3': 100
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


if __name__ == '__main__':
    unittest.main()
