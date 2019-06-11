#! /usr/bin/env python3
"""
A series of functions used to read different spectra file types.

"""
import base64
import itertools
import re
import struct
from typing import Any, Dict, Tuple
import zlib

import numpy as np

import lxml.etree as etree

from .mass_spectrum import Spectrum


MGF_TITLE_REGEX = re.compile(r"Locus:([\d\.]+) ")


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """


def read_mgf_file(spec_file: str) -> Dict[str, Spectrum]:
    """
    Reads the given tandem mass spectrometry data file to extract individual
    spectra.

    Args:
        spec_file (str): The path to the MGF file to read.

    Returns:
        A dictionary of spectrum ID to numpy array of peaks.

    """
    spectra: Dict[str, Spectrum] = {}
    with open(spec_file) as fh:
        for key, group in itertools.groupby(fh, lambda ln: ln == "END IONS\n"):
            if next(group) == "END IONS\n":
                continue

            items = list(group)

            fields = {}
            peak_start_idx = 0
            for ii, line in enumerate(items):
                if "=" not in line:
                    peak_start_idx = ii
                    break
                split_line = line.strip().split("=")
                fields[split_line[0]] = split_line[1]

            match = MGF_TITLE_REGEX.match(fields["TITLE"])
            if match is None:
                raise ParserException("No spectrum ID found in MGF file")
            spec_id = match.group(1)

            spectra[spec_id] = Spectrum(
                np.array([float(n) for s in items[peak_start_idx:]
                          for n in s.split(" ")[:2]]).reshape(-1, 2),
                float(fields["PEPMASS"]),
                int(fields["CHARGE"].split("+")[0]) if "CHARGE" in fields
                else None)

    return spectra


def read_mzml_file(spec_file: str):
    """
    Reads the given mzML file to extract spectra.

    """
    # TODO
    raise NotImplementedError()


def read_mzxml_file(spec_file: str):
    """
    Reads the given mzXML file to extract spectra.

    """
    # TODO
    raise NotImplementedError()


def read_spectra_file(spec_file: str) -> Dict[str, Spectrum]:
    """
    Determines the format of the given tandem mass spectrum file and delegates
    to the appropriate reader.

    Args:
        spec_file (str): The path to the spectrum file to read.

    Returns:

    """
    if spec_file.endswith('.mgf'):
        return read_mgf_file(spec_file)
    if spec_file.lower().endswith('.mzml'):
        return read_mzml_file(spec_file)
    if spec_file.lower().endswith('.mzxml'):
        return read_mzxml_file(spec_file)
    raise NotImplementedError(
        f"Unsupported spectrum file type for {spec_file}")


def decodebinary(string: str, default_array_length: int, precision: int = 64,
                 bzlib: str = 'z') -> Tuple[Any, ...]:
    """
    Decode binary string to float points.
    If provided, should take endian order into consideration.
    """
    decoded = base64.b64decode(string)
    decoded = zlib.decompress(decoded) if bzlib == 'z' else decoded
    unpack_format = "<%dd" % default_array_length if precision == 64 else \
        "<%dL" % default_array_length
    return struct.unpack(unpack_format, decoded)


def mzml_extract_ms1(mzml_file: str,
                     namespace: str = "http://psi.hupo.org/ms/mzml")\
                     -> Dict[str, Dict[str, Any]]:
    """
    Extracts the MS1 spectra from the input mzML file.

    Args:
        msml_file (str): The path to the mzML file.
        namespace (str, optional): The XML namespace used in the mzML file.

    Returns:
        A list of the MS1 spectra encoded in dictionaries.

    """
    spectra = {}
    ns_map = {'x': namespace}
    # read from xml data
    for event, element in etree.iterparse(mzml_file, events=['end']):
        if event == 'end' and element.tag == f"{{{namespace}}}spectrum":
            # This contains the cycle and experiment information
            spectrum_info = dict(element.items())
            default_array_length = int(
                spectrum_info.get('default_array_length',
                                  spectrum_info["defaultArrayLength"]))

            # MS level
            if element.find(f"{{{namespace}}}precursorList"):
                # Ignore MS level >= 2
                continue

            # MS spectrum
            mz_binary = element.xpath(
                "x:binaryDataArrayList/x:binaryDataArray"
                "[x:cvParam[@name='m/z array']]/x:binary",
                namespaces=ns_map)[0]
            int_binary = element.xpath(
                "x:binaryDataArrayList/x:binaryDataArray"
                "[x:cvParam[@name='intensity array']]/x:binary",
                namespaces=ns_map)[0]
            mz = decodebinary(mz_binary.text, default_array_length)
            intensity = decodebinary(int_binary.text, default_array_length)

            # Retention time
            start_time = float(element.xpath(
                "x:scanList/x:scan/x:cvParam[@name='scan start time']",
                namespaces=ns_map)[0].get("value"))

            element.clear()

            # Remove spectral peaks with intensity 0
            mz_intensity = [(mz[ii], intensity[ii])
                            for ii in range(default_array_length)
                            if intensity[ii] > 0]
            mz, intensity = zip(*mz_intensity)

            spec_id = ".".join(re.findall(r"\w+=(\d+)", spectrum_info["id"]))

            spectra[spec_id] = {
                'mz': mz,
                'intensity': intensity,
                'rt': start_time,
                'info': spectrum_info
            }

    return spectra
