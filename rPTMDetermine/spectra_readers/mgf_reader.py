#! /usr/bin/env python3
"""
A module for reading MGF files.

"""
import itertools
import re

import numpy as np

from rPTMDetermine.mass_spectrum import Spectrum

from .core import ParserException, SpectrumGenerator


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
            for key, group in itertools.groupby(
                    fh, lambda ln: ln == "END IONS\n"
            ):
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
