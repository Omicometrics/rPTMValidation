#! /usr/bin/env python3
"""
This module provides a class for reading MS-GF+ mzIdentML files.

"""
import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

from overrides import overrides

from pepfrag import ModSite

from .mzidentml_reader import Ident, MZIdentMLReader, MZIdentMLSearchResult
from .ptmdb import PTMDB
from .search_result import PeptideType


@dataclasses.dataclass
class MSGFPlusSearchResult(MZIdentMLSearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("raw_score", "denovo_score", "spec_evalue", "evalue",)

    raw_score: float
    denovo_score: float
    spec_evalue: float
    evalue: float


class MSGFPlusReader(MZIdentMLReader):  # pylint: disable=too-few-public-methods
    """
    Class to read MS-GF+ mzIdentML files.

    """
    def __init__(self, ptmdb: PTMDB,
                 namespace: str = "http://psidev.info/psi/pi/mzIdentML/1.1"):
        """
        Initialize the reader instance.

        """
        super().__init__(ptmdb, namespace=namespace)

    @overrides
    def _build_search_results(
        self,
        spec_idents: Dict[Tuple[str, Optional[str]], List[Ident]],
        peptides: Dict[str, Tuple[str, List[ModSite]]],
        pep_types: Dict[str, PeptideType]) \
            -> Sequence[MSGFPlusSearchResult]:
        """
        Converts the identifications to MSGFPlusSearchResult objects.

        """
        res: List[MSGFPlusSearchResult] = []
        for (spec_id, dataset), idents in spec_idents.items():
            for ident in idents:
                peptide = peptides[ident.pep_id]

                msgf_scores: Dict[str, float] = {}
                for score in list(ident.scores.keys()):
                    if not score.startswith("MS-GF:"):
                        continue
                    msgf_scores[score[6:]] = \
                        float(ident.scores[score])
                    del ident.scores[score]

                res.append(
                    MSGFPlusSearchResult(
                        seq=peptide[0],
                        mods=peptide[1],
                        charge=ident.charge,
                        spectrum=spec_id,
                        dataset=dataset,
                        rank=ident.rank,
                        pep_type=pep_types[ident.pep_evidence_id],
                        theor_mz=ident.theor_mz,
                        prec_mz=ident.prec_mz,
                        passed_threshold=ident.passed_threshold,
                        scores=ident.scores,
                        raw_score=msgf_scores["RawScore"],
                        denovo_score=msgf_scores["DeNovoScore"],
                        spec_evalue=msgf_scores["SpecEValue"],
                        evalue=msgf_scores["EValue"]))

        return res

    @overrides
    def _extract_spec_idents(self, spec_results) \
            -> Dict[Tuple[str, Optional[str]], List[Ident]]:
        """
        Extracts the peptide identifications for all spectra.

        """
        idents: Dict[Tuple[str, Optional[str]], List[Ident]] = {}
        for spec in spec_results:
            params = self._get_cv_params(spec)
            dataset, spec_id = \
                params["spectrum title"].split(" ")[0].split(":")
            idents[(spec_id, dataset)] = self._extract_spec_peptides(
                spec.findall(self._fix_tag("SpectrumIdentificationItem")))
        return idents
