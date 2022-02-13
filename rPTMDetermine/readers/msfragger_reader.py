"""
This module provides a class for reading MSFragger search results.

"""

import dataclasses
from typing import Any, Dict, Tuple, Optional

from overrides import overrides

from .ptmdb import PTMDB
from .tpp_reader_base import TPPBaseReader, TPPSearchResult
from .search_result import SpectrumIDType


@dataclasses.dataclass(eq=True, frozen=True)
class MSFraggerSearchResult(TPPSearchResult):

    __slots__ = ('massdiff',)

    massdiff: float


class MSFraggerReader(TPPBaseReader):
    """
    Class to read an MSFragger pepXML file.

    """
    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.

        """
        super().__init__(ptmdb)

    @overrides
    def _get_id(self, query_element) -> Tuple[str, SpectrumIDType]:
        """
        Extracts the spectrum ID from the query XML element.

        """
        spec_id = query_element.get('native_id')
        if spec_id is None:
            return query_element.get('start_scan'), SpectrumIDType.scan
        return spec_id, SpectrumIDType.native

    @overrides
    def _extract_hit(self, hit_element, charge: int) -> Dict[str, Any]:
        """
        Parses the search_hit XML element to extract relevant information.

        """
        hit = super()._extract_hit(hit_element, charge)
        hit['massdiff'] = float(hit_element.get('massdiff'))
        return hit

    @staticmethod
    @overrides
    def _build_search_result(spec_id: str,
                             id_type: SpectrumIDType,
                             hit: Dict[str, Any],
                             dataset: Optional[Tuple[str, str]] = None)\
            -> MSFraggerSearchResult:
        """
        Converts a search result to a standard SearchResult.

        Returns:
            MSFraggerSearchResult.

        """
        return MSFraggerSearchResult(
            seq=hit['seq'],
            mods=hit['mods'],
            charge=hit['charge'],
            spectrum=spec_id,
            spectrum_id_type=id_type,
            dataset=dataset,
            rank=hit['rank'],
            pep_type=hit['pep_type'],
            theor_mz=None,
            scores=hit['scores'],
            massdiff=hit['massdiff']
        )
