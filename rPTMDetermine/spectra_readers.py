#! /usr/bin/env python3
"""
A series of functions used to read different spectra file types.

"""
import base64
import collections
import enum
import functools
import itertools
import re
import struct
from typing import Any, Dict, Generator, List, Optional, Tuple
import zlib

import numpy as np

import lxml.etree as etree

from .mass_spectrum import Spectrum


SpectrumGenerator = Generator[Tuple[str, Spectrum], None, None]


MZMLPrecursor = collections.namedtuple(
    "MZMLPrecursor",
    ("prec_id", "selected_ions", "charges", "activation_params"))


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """


class MGFReader:
    """
    """
    locus_regex = re.compile(r"Locus:([\d\.]+)")

    native_regex = re.compile(r".*NativeID:\"(.*)\"")

    def __init__(self):
        """
        """
        self._id_getter = None

    @staticmethod
    def _get_id_locus(text: str) -> str:
        """
        Parses the TITLE line to extract spectrum ID from the Locus field.

        Args:
            text (str): The text of the TITLE line of an MGF block.

        Returns:
            The spectrum identifier.

        Raises:
            ParserException

        """
        match = MGFReader.locus_regex.match(text)
        if match is None:
            raise ParserException("No spectrum ID found using locus pattern")
        return match.group(1)

    @staticmethod
    def _get_id_native(text: str) -> str:
        """
        Parses the TITLE line to extract spectrum ID from the NativeID field.

        Args:
            text (str): The text of the TITLE line of an MGF block.

        Returns:
            The spectrum identifier.

        Raises:
            ParserException

        """
        match = MGFReader.native_regex.match(text)
        if match is None:
            raise ParserException(
                "No spectrum ID found using native ID pattern")
        return ".".join(s.split("=")[1] for s in match.group(1).split(" "))

    def _get_id(self, text: str) -> str:
        """
        """
        if self._id_getter is not None:
            return self._id_getter(text)

        try:
            spec_id = self._get_id_locus(text)
        except ParserException:
            pass
        else:
            self._id_getter = self._get_id_locus
            return spec_id

        try:
            spec_id = self._get_id_native(text)
        except ParserException:
            pass
        else:
            self._id_getter = self._get_id_native
            return spec_id

        raise ParserException("Failed to detect ID in TITLE field")

    def read(self, spec_file: str) -> SpectrumGenerator:
        """
        Reads the given MGF data file to extract spectra.

        Args:
            spec_file (str): The path to the MGF file to read.

        """
        with open(spec_file) as fh:
            for key, group in itertools.groupby(fh, lambda ln: ln == "END IONS\n"):
                if next(group) == "END IONS\n":
                    continue

                items = list(group)

                fields = {}
                for ii, line in enumerate(items):
                    if "=" not in line:
                        peak_start_idx = ii
                        break
                    split_line = line.strip().split("=", maxsplit=1)
                    fields[split_line[0]] = split_line[1]
                else:
                    # No lines containing mz/intensity information found
                    continue

                spec_id = self._get_id(fields["TITLE"])

                pep_mass_floats = fields["PEPMASS"].split(" ")
                pep_mass = float(pep_mass_floats[0])

                spectrum = Spectrum(
                    np.array([float(n) for s in items[peak_start_idx:]
                              for n in s.split(" ")[:2]]).reshape(-1, 2),
                    pep_mass,
                    int(fields["CHARGE"].split("+")[0]) if "CHARGE" in fields
                    else None,
                    retention_time=float(fields["RTINSECONDS"])
                    if "RTINSECONDS" in fields else None)

                yield spec_id, spectrum


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

    def extract_ms1(self, mzml_file: str, **kwargs) -> SpectrumGenerator:
        """
        Extracts the MS1 spectra from the input mzML file.

        Args:
            mzml_file (str): The path to the mzML file.

        """
        return self.extract_msn(mzml_file, 1, **kwargs)

    def extract_ms2(self, mzml_file: str, **kwargs) -> SpectrumGenerator:
        """
        Extracts the MS2 spectra from the input mzML file.

        Args:
            mzml_file (str): The path to the mzML file.

        """
        yield from self.extract_msn(mzml_file, 2, **kwargs)

    def extract_msn(
            self, mzml_file: str, n: int,
            act_method: Optional[str] = None,
            act_energy: Optional[int] = None) -> SpectrumGenerator:
        """
        Extracts the MSn spectra from the input mzML file.

        Args:
            mzml_file (str): The path to the mzML file.
            n (int): The MS level for which to return spectral information.

        """
        # Read from xml data
        context = etree.iterparse(
            mzml_file, events=("end",),
            tag=[self._fix_tag("referenceableParamGroup"),
                 self._fix_tag("spectrum")])
        param_groups: Dict[str, Dict[str, Any]] = {}
        for event, element in context:
            if element.tag == self._fix_tag("referenceableParamGroup"):
                params: Dict[str, Any] = {}
                for param in element.findall(self._fix_tag("cvParam")):
                    params[param.get("name")] = param.get("value", None)
                param_groups[element.get("id")] = params
                continue

            # This contains the cycle and experiment information
            spectrum_info = dict(element.items())
            default_array_length = int(spectrum_info.get(
                'default_array_length', spectrum_info["defaultArrayLength"]))

            # MS level
            try:
                ms_level = int(
                    element.xpath("x:cvParam[@name='ms level']",
                                  namespaces=self.ns_map)[0].get("value"))
            except IndexError:
                group = element.find(
                    self._fix_tag("referenceableParamGroupRef")).get("ref")
                ms_level = int(param_groups[group]["ms level"])
            if ms_level != n:
                continue

            # Extract only the last precursor since this should be the one
            # preceding the current spectrum
            try:
                precursor: Optional[MZMLPrecursor] = \
                    self._extract_precursors(element)[-1]
            except IndexError:
                precursor = None

            # Apply filters based on the activation method/energy if specified
            if (precursor is not None and ms_level >= 2 and
                    (act_method is not None or act_energy is not None)):
                act_params = precursor.activation_params
                if act_method is not None and act_method not in act_params:
                    continue
                if (act_energy is not None and
                        act_params.get("collision energy", None)
                        != act_energy):
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

            # Remove spectral peaks with intensity 0
            mz_intensity: List[Tuple[float, float]] = [
                (mz[ii], intensity[ii])
                for ii in range(default_array_length) if intensity[ii] > 0]

            mz, intensity = (), ()
            if mz_intensity:
                mz, intensity = zip(*mz_intensity)

            spec_id = self._parse_id(spectrum_info["id"])

            element.clear()

            spectrum = Spectrum(
                np.array([mz, intensity]),
                precursor.selected_ions[0] if precursor is not None else None,
                precursor.charges[0] if precursor is not None and
                precursor.charges else None,
                start_time
            )

            yield spec_id, spectrum

    def _extract_precursors(self, spectrum) -> List[MZMLPrecursor]:
        """
        """
        precs: List[MZMLPrecursor] = []
        for precursor in spectrum.xpath("x:precursorList/x:precursor",
                                        namespaces=self.ns_map):
            prec_ref = precursor.get("spectrumRef")
            if prec_ref is None:
                continue
            prec_id = self._parse_id(prec_ref)
            ions = []
            charges = []
            for element in precursor.xpath(
                    "x:selectedIonList/x:selectedIon/x:cvParam",
                    namespaces=self.ns_map):
                if element.get("name") == "selected ion m/z":
                    ions.append(float(element.get("value")))
                elif element.get("name") == "charge state":
                    charges.append(int(element.get("value")))
            act_params = {e.get("name"): e.get("value")
                          for e in precursor.xpath("x:activation/x:cvParam",
                                                   namespaces=self.ns_map)}
            if "collision energy" in act_params:
                act_params["collision energy"] = \
                    float(act_params["collision energy"])
            precs.append(MZMLPrecursor(prec_id, ions, charges, act_params))
        return precs

    @staticmethod
    def _parse_id(id_str: str) -> str:
        """
        Parses the ID string to extract the numeric elements.

        """
        return ".".join(re.findall(r"\w+=(\d+)", id_str))

    @staticmethod
    def decode_binary(string: str, default_array_length: int,
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
        if string is None:
            return ()
        decoded = base64.b64decode(string)
        if comp_mode is CompressionMode.zlib:
            decoded = zlib.decompress(decoded)
        unpack_format = "<%dd" % default_array_length if precision == 64 else \
            "<%df" % default_array_length
        return struct.unpack(unpack_format, decoded)

    def _process_binary_data_array(self, data_array,
                                   default_array_length: int) \
            -> Tuple[Any, ...]:
        """
        Processes the binary data array to extract the binary content.

        """
        params = {e.get("name")
                  for e in data_array.findall(self._fix_tag("cvParam"))}
        return self.decode_binary(
            data_array.find(self._fix_tag("binary")).text,
            default_array_length,
            precision=64 if "64-bit float" in params else 32,
            comp_mode=CompressionMode.zlib if "zlib compression" in params
            else None)

    @functools.lru_cache(maxsize=128)
    def _fix_tag(self, tag):
        """
        Prepends the namespace to the tag.

        """
        return f"{{{self.namespace}}}{tag}"


def read_spectra_file(spec_file: str, **kwargs) -> SpectrumGenerator:
    """
    Determines the format of the given tandem mass spectrum file and delegates
    to the appropriate reader.

    Args:
        spec_file (str): The path to the spectrum file to read.

    Returns:

    """
    if spec_file.endswith('.mgf'):
        yield from MGFReader().read(spec_file)
    elif spec_file.lower().endswith('.mzml'):
        yield from MZMLReader().extract_ms2(spec_file, **kwargs)
    else:
        raise NotImplementedError(
            f"Unsupported spectrum file type for {spec_file}")
