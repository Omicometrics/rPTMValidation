from .core import ParserException, SpectrumGenerator
from .mgf_reader import MGFReader
from .mzml_reader import MZMLReader, MZMLPrecursor


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


__all__ = [
    'MGFReader',
    'MZMLReader',
    'MZMLPrecursor',
    'ParserException',
    'SpectrumGenerator',
    'read_spectra_file',
]
