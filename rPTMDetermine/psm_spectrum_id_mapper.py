"""
This module converts spectral ID in obtained from mass spectral files
(e.g., .mgf or .mzML) to the mass spectral ID recorded in peptide
identification results.
"""

from .readers import SearchEngine


class SpectrumIDMapper:
    """ A class maps spectrum ID. """

    def __init__(self, search_engine: SearchEngine, file_type: str):
        self.engine = search_engine
        self.spectrum_file_type = file_type
        self._converter = None

    def convert(self, spectrum_id: str) -> str:
        """
        Converts spectrum ID.

        Args:
            spectrum_id: input spectrum ID

        Returns:
            Converted spectrum ID
        """
        if self._converter is not None:
            return self._converter(spectrum_id)

        return self._convert(spectrum_id)

    def _convert(self, spectrum_id: str) -> str:
        """ Converter method

        Args:
            spectrum_id: spectral ID

        Returns:

        """
        if self.engine:
            pass
