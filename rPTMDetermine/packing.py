"""
Module to provide packing and unpacking using MessagePack.

Numpy parts of this code have been extracted and adapted from msgpack_numpy
for flexibility.

License for code from msgpack_numpy:
Copyright (c) 2013-2020, Lev E. Givon
All rights reserved.
Distributed under the terms of the BSD license:
http://www.opensource.org/licenses/bsd-license

"""
import dataclasses
import enum
import inspect
import importlib
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Union

import msgpack
import numpy as np


def fullname(obj):
    """
    Gets the full name, including module, of the `obj`.

    Adapted from https://stackoverflow.com/questions/2020014.

    """
    if isinstance(obj, type) or isinstance(obj, FunctionType):
        # A class itself, not an instance
        return f'{obj.__module__}.{obj.__name__}'

    return f'{obj.__class__.__module__}.{obj.__class__.__name__}'


def ndarray_to_bytes_darwin(obj):
    return obj.tobytes()


def ndarray_to_bytes_other(obj):
    return obj.data if obj.flags['C_CONTIGUOUS'] else obj.tobytes()


ndarray_to_bytes = (
    ndarray_to_bytes_darwin
    if sys.platform == 'darwin'
    else ndarray_to_bytes_other
)


def num_to_bytes(obj: Union[np.bool_, np.number]):
    return obj.data


def tostr(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _encode(obj):
    if isinstance(obj, enum.Enum):
        return {
            b'class': b'Enum',
            b'subclass': fullname(obj),
            b'value': obj.value
        }

    if isinstance(obj, set):
        # Store a set as a list
        return {
            b'class': b'set',
            b'data': list(obj)
        }

    if isinstance(obj, MappingProxyType):
        return {
            b'class': b'MappingProxyType',
            b'data': {k: v for k, v in obj.items()}
        }

    if isinstance(obj, FunctionType):
        return {
            b'function': fullname(obj)
        }

    # numpy - adapted from msgpack_numpy
    if isinstance(obj, np.ndarray):
        # If the dtype is structured, store the interface description;
        # otherwise, store the corresponding array protocol type string:
        if obj.dtype.kind == 'V':
            kind = b'V'
            descr = obj.dtype.descr
        else:
            kind = b''
            descr = obj.dtype.str

        return {
            b'nd': True,
            b'type': descr,
            b'kind': kind,
            b'shape': obj.shape,
            # Modified by Daniel Spencer (17/6/2020)
            b'data': obj.tolist() if descr == '|O' else ndarray_to_bytes(obj)
        }

    if isinstance(obj, (np.bool_, np.number)):
        return {
            b'nd': False,
            b'type': obj.dtype.str,
            b'data': num_to_bytes(obj)
        }

    if isinstance(obj, complex):
        return {
            b'complex': True,
            b'data': obj.__repr__()
        }

    if isinstance(obj, object):
        if obj.__class__.__name__ == 'type':
            # Not an instance of a class, but the class itself
            return {b'class': b'type', b'typename': fullname(obj)}

        d = {b'class': fullname(obj)}
        try:
            d[b'data'] = obj.__dict__
        except AttributeError:
            # Since classes with slots do not have direct access to the
            # parent classes' slots, inspect is used to use the method
            # resolution order to retrieved these
            slots = []
            for base_class in inspect.getmro(obj.__class__):
                try:
                    slots.extend(getattr(base_class, '__slots__'))
                except AttributeError:
                    continue
            d[b'data'] = {
                slot: getattr(obj, slot) for slot in slots if hasattr(obj, slot)
            }
        return d

    return obj


def _decode(obj):
    if b'function' in obj:
        module, func_name = obj[b'function'].rsplit('.', maxsplit=1)
        return getattr(importlib.import_module(module), func_name)

    if b'nd' in obj:
        if obj[b'nd'] is True:
            # Check if b'kind' is in obj to enable decoding of data
            # serialized with older versions (#20):
            if b'kind' in obj and obj[b'kind'] == b'V':
                descr = [tuple(tostr(t) if type(t) is bytes else t for t in d)
                         for d in obj[b'type']]
            else:
                descr = obj[b'type']

            # Added by Daniel Spencer (17/6/2020)
            if descr == '|O':
                return np.array(
                    obj[b'data'], dtype=np.dtype(descr)
                ).reshape(obj[b'shape'])

            # A copy of the array is returned to ensure that the array is
            # writable
            return np.frombuffer(
                obj[b'data'], dtype=np.dtype(descr)
            ).reshape(obj[b'shape']).copy()
        else:
            descr = obj[b'type']
            return np.frombuffer(
                obj[b'data'], dtype=np.dtype(descr)
            )[0].copy()

    if b'complex' in obj:
        return complex(tostr(obj[b'data']))

    if b'class' in obj:
        if obj[b'class'] == b'set':
            return set(obj[b'data'])

        if obj[b'class'] == b'type':
            module, type_name = obj[b'typename'].rsplit('.', maxsplit=1)
            return getattr(importlib.import_module(module), type_name)

        if obj[b'class'] == b'Enum':
            module, type_name = obj[b'subclass'].rsplit('.', maxsplit=1)
            cls = getattr(importlib.import_module(module), type_name)
            return cls(obj[b'value'])

        if obj[b'class'] == b'MappingProxyType':
            return MappingProxyType(obj[b'data'])

        module, type_name = obj[b'class'].rsplit('.', maxsplit=1)
        cls = getattr(importlib.import_module(module), type_name)
        if cls is not None:
            if dataclasses.is_dataclass(cls):
                try:
                    # Reconstruct dataclass by initialization
                    return cls(**obj[b'data'])
                except TypeError:
                    # No suitable __init__ constructor
                    new_obj = cls.__new__(cls)
                    for slot in new_obj.__slots__:
                        setattr(new_obj, slot, obj[b'data'][slot])

            try:
                new_obj = cls.__new__(cls)
            except TypeError as ex:
                raise TypeError(
                    f'Error while unpacking {module}.{type_name}: {ex}'
                )
            try:
                new_obj.__dict__ = obj[b'data']
            except AttributeError:
                for slot in new_obj.__slots__:
                    setattr(new_obj, slot, obj[b'data'][slot])
            return new_obj

        raise TypeError(f'Class {obj[b"class"]} is not supported')

    return obj


def save_to_file(obj: Any, file_path: str):
    with open(file_path, 'wb') as fh:
        fh.write(msgpack.packb(obj, default=_encode))


def load_from_file(file_path: str, use_list: bool = False) -> Any:
    with open(file_path, 'rb') as fh:
        return msgpack.unpackb(
            fh.read(),
            object_hook=_decode,
            use_list=use_list,
            strict_map_key=False
        )
