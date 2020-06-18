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
import functools
import inspect
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Dict, Optional, Type, Union

import msgpack
import numpy as np
from pepfrag import MassType, ModSite, Peptide
from sklearn.metrics._scorer import _passthrough_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from .features import Features
from .machinelearning import classification, stacking
from .mass_spectrum import Spectrum
from .readers import (
    MascotSearchResult,
    MSGFPlusSearchResult,
    MZIdentMLSearchResult,
    PeptideType,
    PercolatorSearchResult,
    ProteinPilotSearchResult,
    ProteinPilotXMLSearchResult,
    SearchResult,
    TPPSearchResult
)
from .peptide_spectrum_match import PSM
from .psm_container import PSMContainer
from .validation_model import Scaler, ValidationModel


SUPPORTED_CLASSES: Dict[str, Type[object]] = {
    'pepfrag.constants.MassType': MassType,
    'pepfrag.pepfrag.ModSite': ModSite,
    'pepfrag.pepfrag.Peptide': Peptide,
    'rPTMDetermine.features.Features': Features,
    'rPTMDetermine.readers.search_result.PeptideType': PeptideType,
    'rPTMDetermine.peptide_spectrum_match.PSM': PSM,
    'rPTMDetermine.psm_container.PSMContainer': PSMContainer,
    'rPTMDetermine.mass_spectrum.Spectrum': Spectrum,
    # Machine Learning
    'rPTMDetermine.validation_model.Scaler': Scaler,
    'rPTMDetermine.validation_model.ValidationModel': ValidationModel,
    'rPTMDetermine.machinelearning.classification.Classifier': classification.Classifier,
    'rPTMDetermine.machinelearning.classification.CrossValidationClassifier': classification.CrossValidationClassifier,
    'rPTMDetermine.machinelearning.classification.EnsembleClassifier': classification.EnsembleClassifier,
    'rPTMDetermine.machinelearning.classification.Model': classification.Model,
    'rPTMDetermine.machinelearning.classification.ModelMetrics': classification.ModelMetrics,
    'rPTMDetermine.machinelearning.stacking.Model': stacking.Model,
    'rPTMDetermine.machinelearning.stacking.Stacking': stacking.Stacking,
    'rPTMDetermine.machinelearning.stacking.Weight': stacking.Weight,
    'sklearn.model_selection._search.GridSearchCV': GridSearchCV,
    'sklearn.svm._classes.LinearSVC': LinearSVC,
    'sklearn.preprocessing._data.MaxAbsScaler': MaxAbsScaler,
    # SearchResults
    'rPTMDetermine.readers.mascot_reader.MascotSearchResult': MascotSearchResult,
    'rPTMDetermine.readers.msgfplus_reader.MSGFPlusSearchResult': MSGFPlusSearchResult,
    'rPTMDetermine.readers.mzidentml_reader.MZIdentMLSearchResult': MZIdentMLSearchResult,
    'rPTMDetermine.readers.percolator_reader.PercolatorSearchResult': PercolatorSearchResult,
    'rPTMDetermine.readers.protein_pilot_reader.ProteinPilotSearchResult': ProteinPilotSearchResult,
    'rPTMDetermine.readers.protein_pilot_reader.ProteinPilotXMLSearchResult': ProteinPilotXMLSearchResult,
    'rPTMDetermine.readers.search_result.SearchResult': SearchResult,
    'rPTMDetermine.readers.tpp_reader.TPPSearchResult': TPPSearchResult
}


SUPPORTED_FUNCTIONS: Dict[str, Any] = {
    'sklearn.metrics._scorer._passthrough_scorer': _passthrough_scorer,
}


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
            d[b'data'] = {slot: getattr(obj, slot) for slot in slots}
        return d

    return obj


def _decode(
        obj,
        extra_classes: Optional[Dict[str, Type[object]]],
        extra_functions: Optional[Dict[str, FunctionType]]
):
    if b'function' in obj:
        func = {
            **SUPPORTED_FUNCTIONS, **(extra_functions or {})
        }.get(obj[b'function'])
        if func is not None:
            return func
        raise TypeError(f'Function {obj[b"function"]} is not supported')

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

        supported_classes = {**SUPPORTED_CLASSES, **(extra_classes or {})}

        if obj[b'class'] == b'type':
            cls = supported_classes.get(obj[b'typename'])
            if cls is not None:
                return cls
            raise TypeError(f'Class {obj[b"typename"]} is not supported')

        if obj[b'class'] == b'Enum':
            cls = supported_classes.get(obj[b'subclass'])
            if cls is not None:
                # Re-construct Enum by initialization
                return cls(obj[b'value'])
            raise TypeError(f'Class {obj[b"subclass"]} is not supported')

        if obj[b'class'] == b'MappingProxyType':
            return MappingProxyType(obj[b'data'])

        cls = supported_classes.get(obj[b'class'])
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

            new_obj = cls.__new__(cls)
            try:
                new_obj.__dict__ = obj[b'data']
            except AttributeError:
                for slot in new_obj.__slots__:
                    setattr(new_obj, slot, obj[b'data'][slot])
            return new_obj

        raise TypeError(f'Class {obj[b"class"]} is not supported')

    return obj


def save_to_file(
        obj: Any,
        file_path: str
):
    with open(file_path, 'wb') as fh:
        fh.write(msgpack.packb(obj, default=_encode))


def load_from_file(
        file_path: str,
        extra_classes: Optional[Dict[str, Type[object]]] = None,
        extra_functions: Optional[Dict[str, FunctionType]] = None,
        use_list: bool = False
) -> Any:
    with open(file_path, 'rb') as fh:
        return msgpack.unpackb(
            fh.read(),
            object_hook=functools.partial(
                _decode,
                extra_classes=extra_classes,
                extra_functions=extra_functions
            ),
            use_list=use_list,
            strict_map_key=False
        )
