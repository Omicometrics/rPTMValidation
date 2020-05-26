#! /usr/bin/env python3
"""
A module to define a base `Config` class for easy generation of
configuration classes.

"""
import collections
import enum
import hashlib
import json
from typing import Any, Dict, List, Tuple


class MissingConfigOptionException(Exception):
    """
    An exception to be thrown when a required configuration option is missing.

    """


ConfigField = collections.namedtuple(
    'ConfigField',
    [
        'name',
        'has_default',
        'default',
        'caster',
    ],
    defaults=[None, False, lambda v: v]
)


def make_getter(field: ConfigField):
    def fget(self):
        if field.has_default:
            if isinstance(field.default, ConfigField):
                try:
                    value = self.json_config[field.name]
                except KeyError:
                    return getattr(self, field.default.name)
                return field.caster(value)
            return field.caster(self.json_config.get(field.name, field.default))
        else:
            return field.caster(self.json_config[field.name])
    return fget


class ConfigMeta(type):
    """
    This metaclass is used to automatically generate properties with 'getter'
    methods for the configured `config_fields`. The `ConfigField` entries
    may optionally specify a default value (by setting `has_default` and
    `default`) to be used if the key was not provided in the input
    configuration. A `ConfigField` may also be used as the default value; in
    this case, the getter of the associated property will be used as a fallback.

    """
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        for field in attrs['config_fields']:
            setattr(
                cls,
                field.name,
                property(make_getter(field))
            )


class Config(metaclass=ConfigMeta):
    """
    This class provides a common interface for configuration files, enabling
    centralized options and default values.

    Configuration properties are constructed by the metaclass `ConfigMeta`,
    using the `config_fields` class variable.

    @DynamicAttrs
    """
    config_fields: List[ConfigField] = []

    def __init__(self, json_config: Dict[str, Any]):
        """
        Initialize the class using the JSON configuration.

        Raises:
            MissingConfigOptionException.

        """
        self.json_config = json_config
        self.required = [f.name for f in self.config_fields
                         if not f.has_default]
        self._check_required()

    def __str__(self) -> str:
        """
        Implements the string conversion for the class.

        """
        string = ''
        for field in self.config_fields:
            val = getattr(self, field.name)
            if isinstance(val, enum.Enum):
                val = val.name
            string += f'{field.name} = {str(val)}\n'
        return string

    def __hash__(self):
        """
        Implements the hash special method for the class.

        """
        attrs = []
        for field in self.config_fields:
            attr = getattr(self, field.name)
            if isinstance(attr, dict):
                attr = json.dumps(attr, sort_keys=True, default=str)
            elif isinstance(attr, list):
                attr = tuple(attr)
            attrs.append(attr)
        return int(hashlib.sha1(str(attrs).encode('utf8')).hexdigest(), 16)

    def __eq__(self, other):
        """
        Implements the equality test for the class.

        """
        if isinstance(other, Config):
            return self.__value_tuple() == other.__value_tuple()
        return NotImplemented

    def __value_tuple(self) -> Tuple:
        """
        Constructs a tuple of the ConfigField values for the class.

        Returns:
            Tuple of field values.

        """
        return tuple([getattr(self, f.name) for f in self.config_fields])

    def _check_required(self):
        """
        Checks that the required options have been set in the configuration
        file.

        Raises:
            MissingConfigOptionException.

        """
        for attr in self.required:
            try:
                getattr(self, attr)
            except KeyError as ex:
                raise MissingConfigOptionException(
                    f"Missing required config option: {attr}") from ex
