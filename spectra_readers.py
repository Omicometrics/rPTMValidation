#! /usr/bin/env python3
"""
A series of functions used to read different spectra file types.

"""
import base64
import re
import struct
import zlib

import lxml.etree as etree

import mass_spectrum


MGF_TITLE_REGEX = re.compile(r"TITLE=Locus:([\d\.]+) ")


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """
    
    
END_IONS_LEN = 8
TITLE_LEN = 5
PEPMASS_LEN = 7
BEGIN_IONS_LEN = 10
CHARGE_LEN = 6


def read_mgf_file(spec_file):
    """
    Reads the given tandem mass spectrometry data file to extract individual
    spectra.

    Args:
        spec_file (str): The path to the MGF file to read.

    Returns:
        A dictionary of spectrum ID to numpy array of peaks.

    """
    spectra = {}
    spec_id = None    
    with open(spec_file) as fh:
        peaks, mz, charge = [], None, None
        for line in fh:
            if line[:END_IONS_LEN] == "END IONS":
                if spec_id is None:
                    raise ParserException(
                        f"No spectrum ID found in MGF block in {spec_file}")
                spectra[spec_id] = mass_spectrum.Spectrum(peaks, float(mz), charge)
                peaks, spec_id, mz, charge = [], None, None, None
            elif line[:TITLE_LEN] == "TITLE":
                spec_id = MGF_TITLE_REGEX.match(line).group(1)
            elif line[:PEPMASS_LEN] == "PEPMASS":
                mz = line.rstrip().split("=")[1]
            elif line[:CHARGE_LEN] == "CHARGE":
                charge = line.rstrip().split("=")[1]
            elif "=" not in line and line[:BEGIN_IONS_LEN] != "BEGIN IONS":
                peaks.append([float(n) for n in line.split()[:2]])

    return spectra


def read_mzml_file(spec_file):
    """
    Reads the given mzML file to extract spectra.

    """
    # TODO
    raise NotImplementedError()


def read_mzxml_file(spec_file):
    """
    Reads the given mzXML file to extract spectra.

    """
    # TODO
    raise NotImplementedError()


def read_spectra_file(spec_file):
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


def decodebinary(string, default_array_length, precision=64, bzlib='z'):
    """
    Decode binary string to float points.
    If provided, should take endian order into consideration.
    """
    decoded = base64.b64decode(string)
    decoded = zlib.decompress(decoded) if bzlib == 'z' else decoded
    unpack_format = "<%dd" % default_array_length if precision == 64 else \
        "<%dL" % default_array_length
    return struct.unpack(unpack_format, decoded)


def mzml_extract_ms1(mzml_file, namespace="http://psi.hupo.org/ms/mzml"):
    """
    Extracts the MS1 spectra from the input mzML file.

    Args:
        msml_file (str): The path to the mzML file.
        namespace (str, optional): The XML namespace used in the mzML file.

    Returns:
        A list of the MS1 spectra encoded in dictionaries.

    """
    spectra = []
    ns_map = {'x': namespace}
    # read from xml data
    for event, element in etree.iterparse(mzml_file, events=['end']):
        if event == 'end' and element.tag == f"{{{namespace}}}spectrum":
            # This contains the cycle and experiment information
            spectrum_info = dict(element.items())
            default_array_length = int(spectrum_info['default_array_length'])

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

            spectra.append({
                'mz': mz,
                'intensity': intensity,
                'rt': start_time,
                'info': spectrum_info
            })

    return spectra
