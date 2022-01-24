"""
This module provides a class for parsing TPP pepXML files.

"""
import dataclasses

from typing import Any, Dict, Optional, Tuple
from overrides import overrides

from .ptmdb import PTMDB
from .tpp_reader_base import TPPBaseReader, TPPSearchResult


@dataclasses.dataclass(eq=True, frozen=True)
class TPPSearchResult(TPPSearchResult):

    __slots__ = ("pprophet_prob",
                 "pprophet_params",
                 "iprophet_prob",
                 "iprophet_params")

    pprophet_prob: Optional[float]
    pprophet_params: Optional[Dict[str, float]]
    iprophet_prob: Optional[float]
    iprophet_params: Optional[Dict[str, float]]


class TPPReader(TPPBaseReader):
    """
    Class to read a TPP pep.xml file.

    """
    def __init__(self, ptmdb: PTMDB):
        """
        Initialize the reader.

        Args:
            ptmdb (PTMDB): The UniMod PTM database.

        """
        super().__init__(ptmdb)

    @overrides
    def _extract_hit(self, hit_element, charge: int) -> Dict[str, Any]:
        """
        Parses the search_hit XML element to extract relevant information.

        """
        hit = super()._extract_hit(hit_element, charge)

        # PeptideProphet post-processed search results
        pprophet_probs, iprophet_probs = self._process_prophet(hit_element)
        if pprophet_probs:
            hit['pprophet_prob'] = pprophet_probs["prob"]
            hit['pprophet_params'] = pprophet_probs["param"]
        else:
            hit['pprophet_prob'] = None
            hit['pprophet_params'] = None

        if iprophet_probs:
            hit['iprophet_prob'] = pprophet_probs["prob"]
            hit['iprophet_params'] = pprophet_probs["param"]
        else:
            hit['iprophet_prob'] = None
            hit['iprophet_params'] = None
        return hit

    def _process_prophet(self, elem)\
            -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Parse the search_hit XML element to extract PeptideProphet and
        iProphet parameters and probability.

        """
        pprophet_probs: Dict[str, float] = {}
        iprophet_probs: Dict[str, float] = {}
        for anal in elem.xpath("x:analysis_result", namespaces=self.ns_map):
            scores = {s.get("name"): float(s.get("value"))
                      for s in anal.xpath(".//x:parameter",
                                          namespaces=self.ns_map)}
            if anal.get("analysis") == "peptideprophet":
                for p in anal.xpath("x:peptideprophet_result",
                                    namespaces=self.ns_map):
                    pprophet_probs["prob"] = float(p.get("probability"))
                pprophet_probs["param"] = scores
            elif anal.get("analysis") == "interprophet":
                for p in anal.xpath("x:interprophet_result",
                                    namespaces=self.ns_map):
                    iprophet_probs["prob"] = float(p.get("probability"))
                iprophet_probs["param"] = scores
        return pprophet_probs, iprophet_probs

    @staticmethod
    def _build_search_result(
            raw_file: str,
            scan_no: int,
            spec_id: str,
            hit: Dict[str, Any]
    ) -> TPPSearchResult:
        """
        Converts a search result to a standard SearchResult.

        Returns:
            TPPSearchResult.

        """
        return TPPSearchResult(
            seq=hit['seq'],
            mods=hit['mods'],
            charge=hit['charge'],
            spectrum=spec_id,
            dataset=None,
            rank=hit['rank'],
            pep_type=hit['pep_type'],
            theor_mz=None,
            scores=hit['scores'],
            pprophet_prob=hit['pprophet_prob'],
            pprophet_params=hit['pprophet_params'],
            iprophet_prob=hit['iprophet_prob'],
            iprophet_params=hit['iprophet_params'],
        )
