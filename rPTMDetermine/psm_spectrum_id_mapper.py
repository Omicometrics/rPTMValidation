"""
This module converts spectral ID in obtained from mass spectral files
(e.g., .mgf or .mzML) to the mass spectral ID recorded in peptide
identification results.
"""
import re

from .readers import SearchEngine, SpectrumIDType


class SpectrumIDMapper:
    """ A class maps spectrum ID. """
    native_regex = re.compile(r".*NativeID:\"(.*)\"")

    def __init__(self,
                 search_engine: SearchEngine,
                 file_type: str,
                 id_type: SpectrumIDType):
        self.engine = search_engine
        self.spectrum_file_type = file_type
        self.spectrum_id_type = id_type
        self._converter = None
        self._create_converter()

    def convert(self, spectrum_id: str) -> str:
        """
        Converts spectrum ID.

        Args:
            spectrum_id: input spectrum ID

        Returns:
            Converted spectrum ID
        """
        return self._converter(spectrum_id)

    def _create_converter(self):
        """ Converter method

        Args:
            spectrum_id: spectral ID

        Returns:

        """
        if (self.engine in (SearchEngine.Comet,
                            SearchEngine.Mascot,
                            SearchEngine.MSFragger,
                            SearchEngine.TPP)
                and self.spectrum_file_type in ("mgf", "mzML")):
            self._converter = lambda x: x

            if self.spectrum_id_type == "scan_num":
                if self.spectrum_file_type == "mgf":
                    self._converter = self._get_native_scan_num
                elif self.spectrum_file_type == "mzML":
                    self._converter = self._get_mzml_scan_num
                else:
                    raise ValueError("Unrecognized mass spectrum file type: "
                                     f"{self.spectrum_file_type}")
            return

        if self.engine in (SearchEngine.ProteinPilotXML,
                           SearchEngine.ProteinPilot):
            if self.spectrum_file_type == "mgf":
                self._converter = lambda x: x.split()[0].split(":")[1]

            raise NotImplementedError(
                "Mapping the mass spectrum file with type "
                f"{self.spectrum_file_type} to ProteinPilot search results "
                "is not currently implemented. Export .mgf spectrum using "
                "ProteinPilot software instead."
            )

    @staticmethod
    def _get_native_scan_num(spectrum_id: str):
        """
        Parses spectrum ID to get scan number
        """
        match = SpectrumIDMapper.native_regex.match(spectrum_id)
        spec_id_dict = dict(re.findall(r"(\w+)=(\d+)", match.group(1)))
        return spec_id_dict.get("scan")

    @staticmethod
    def _get_mzml_scan_num(spectrum_id: str):
        """
        Parses spectrum ID to get scan number: mzML
        """
        spec_id_dict = dict(re.findall(r"(\w+)=(\d+)", spectrum_id))
        return spec_id_dict.get("scan")
