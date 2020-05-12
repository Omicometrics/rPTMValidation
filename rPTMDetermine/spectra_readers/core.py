from typing import Generator, Tuple

from rPTMDetermine.mass_spectrum import Spectrum


SpectrumGenerator = Generator[Tuple[str, Spectrum], None, None]


class ParserException(Exception):
    """
    A custom exception to be raised during file parse errors.

    """
