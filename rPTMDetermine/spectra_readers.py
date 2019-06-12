#! /usr/bin/env python3
"""
A series of functions used to read different spectra file types.

"""
import base64
import enum
import itertools
import re
import struct
from typing import Any, Dict, Optional, Tuple
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


class CompressionMode(enum.Enum):
    zlib = enum.auto()


class MZMLReader:
    """
    A reader class for mzML files.

    """
    def __init__(self, namespace: str = "http://psi.hupo.org/ms/mzml"):
        """
        Initializes the MZMLReader.

        Args:
            namespace (str, optional): The XML namespace used in the mzML file.

        """
        self.namespace = namespace
        self.ns_map = {'x': self.namespace}

    def extract_ms1(self, mzml_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Extracts the MS1 spectra from the input mzML file.

        Args:
            msml_file (str): The path to the mzML file.

        Returns:
            A list of the MS1 spectra encoded in dictionaries.

        """
        spectra = {}
        # read from xml data
        for event, element in etree.iterparse(mzml_file, events=['end']):
            if (event == 'end' and
                    element.tag == f"{{{self.namespace}}}spectrum"):
                # This contains the cycle and experiment information
                spectrum_info = dict(element.items())
                default_array_length = int(
                    spectrum_info.get('default_array_length',
                                      spectrum_info["defaultArrayLength"]))

                # MS level
                if element.find(f"{{{self.namespace}}}precursorList"):
                    # Ignore MS level >= 2
                    continue

                # MS spectrum
                def _get_array(s):
                    return element.xpath(
                        "x:binaryDataArrayList/x:binaryDataArray"
                        f"[x:cvParam[@name='{s} array']]",
                        namespaces=self.ns_map)[0]

                mz = self._process_binary_data_array(
                    _get_array("m/z"), default_array_length)
                intensity = self._process_binary_data_array(
                    _get_array("intensity"), default_array_length)

                # Retention time
                start_time = float(element.xpath(
                    "x:scanList/x:scan/x:cvParam[@name='scan start time']",
                    namespaces=self.ns_map)[0].get("value"))

                element.clear()

                # Remove spectral peaks with intensity 0
                mz_intensity = [(mz[ii], intensity[ii])
                                for ii in range(default_array_length)
                                if intensity[ii] > 0]
                mz, intensity = zip(*mz_intensity)

                spec_id = ".".join(
                    re.findall(r"\w+=(\d+)", spectrum_info["id"]))

                spectra[spec_id] = {
                    'mz': mz,
                    'intensity': intensity,
                    'rt': start_time,
                    'info': spectrum_info
                }

        return spectra

    def decode_binary(self, string: str, default_array_length: int,
                      precision: int = 64,
                      comp_mode: Optional[CompressionMode] = None) \
            -> Tuple[Any, ...]:
        """
        Decodes binary string to floats.

        Args:
            string (str): The binary content as a string.
            default_array_length (int): The default array length, read from
                                        the mzML file.
            precision (int, optional): The precision of the binary data.
            comp_mode (CompressionMode, optional): The compression mode.

        Returns:
            The decoded and decompressed binary content as a tuple.

        """
        decoded = base64.b64decode(string)
        if comp_mode is CompressionMode.zlib:
            decoded = zlib.decompress(decoded)
        unpack_format = "<%dd" % default_array_length if precision == 64 else \
            "<%dL" % default_array_length
        return struct.unpack(unpack_format, decoded)

    def _process_binary_data_array(self, data_array,
                                   default_array_length: int) \
            -> Tuple[Any, ...]:
        """
        Processes the binary data array to extract the binary content.

        """
        params = {e.get("name")
                  for e in data_array.findall(f"{{{self.namespace}}}cvParam")}
        return self.decode_binary(
            data_array.find(f"{{{self.namespace}}}binary").text,
            default_array_length,
            precision=64 if "64-bit float" in params else 32,
            comp_mode=CompressionMode.zlib if "zlib compression" in params
            else None)
