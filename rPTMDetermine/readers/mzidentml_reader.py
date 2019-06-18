#! /usr/bin/env python3
"""
This module provides a class for reading MS-GF+ mzIdentML files.

"""
import collections
import dataclasses
import html
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import lxml.etree as etree
from overrides import overrides

from pepfrag import ModSite

from .base_reader import Reader
from .ptmdb import PTMDB
from .search_result import PeptideType, SearchResult


Ident = collections.namedtuple("Ident",
                               ["pep_id", "pep_evidence_id", "charge", "rank",
                                "theor_mz", "prec_mz", "passed_threshold",
                                "scores"])


@dataclasses.dataclass
class MZIdentMLSearchResult(SearchResult):  # pylint: disable=too-few-public-methods

    __slots__ = ("prec_mz", "passed_threshold", "scores",)

    prec_mz: float
    passed_threshold: bool
    scores: Dict[str, float]


class MZIdentMLReader(Reader):  # pylint: disable=too-few-public-methods
    """
    Class to read mzIdentML files.

    """
    def __init__(self, ptmdb: PTMDB,
                 namespace: str = "http://psidev.info/psi/pi/mzIdentML/1.2"):
        """
        Initialize the reader instance.

        """
        super().__init__(ptmdb)

        self.namespace = namespace
        self.ns_map = {'x': self.namespace}

        self._seq_collection_tag = self._fix_tag("SequenceCollection")
        self._spec_ident_list_tag = self._fix_tag("SpectrumIdentificationList")

    @overrides
    def read(self, filename: str,
             predicate: Optional[Callable[[SearchResult], bool]] = None,
             **kwargs) -> Sequence[SearchResult]:
        """
        Reads the given mzIdentML result file.

        Args:
            filename (str): The path to the MS-GF+ database search results
                            file.
            predicate (Callable, optional): An optional predicate to filter
                                            results.

        Returns:

        """
        pep_types: Dict[str, PeptideType] = {}
        peptides: Dict[str, Tuple[str, List[ModSite]]] = {}
        spec_idents: Dict[Tuple[str, Optional[str]], List[Ident]] = {}
        context = etree.iterparse(
            filename, events=("end",),
            tag=(self._seq_collection_tag, self._spec_ident_list_tag,))
        for event, element in context:
            if element.tag == self._seq_collection_tag:
                pep_types = self._extract_peptide_types(
                    element.findall(self._fix_tag("PeptideEvidence")))

                peptides = self._extract_peptides(
                    element.findall(self._fix_tag("Peptide")))
            elif element.tag == self._spec_ident_list_tag:
                spec_idents = self._extract_spec_idents(
                    element.findall(
                        self._fix_tag("SpectrumIdentificationResult")))

            element.clear()

        res = self._build_search_results(spec_idents, peptides, pep_types)

        return res if not predicate else [r for r in res if predicate(r)]

    def _build_search_results(
        self,
        spec_idents: Dict[Tuple[str, Optional[str]], List[Ident]],
        peptides: Dict[str, Tuple[str, List[ModSite]]],
        pep_types: Dict[str, PeptideType]) \
            -> Sequence[MZIdentMLSearchResult]:
        """
        Converts the identifications to MZIdentMLSearchResult objects.

        """
        res: List[MZIdentMLSearchResult] = []
        for (spec_id, dataset), idents in spec_idents.items():
            for ident in idents:
                peptide = peptides[ident.pep_id]
                res.append(
                    MZIdentMLSearchResult(
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
                        scores=ident.scores))

        return res

    def _extract_peptide_types(self, pep_evidences) -> Dict[str, PeptideType]:
        """
        Extracts the peptide types (normal or decoy) from the PeptideEvidence
        elements.

        Args:
            pep_evidences: Iterable of eptideEvidence XML elements.

        Returns:
            A dictionary mapping peptide evidence ID to PeptideType.

        """
        return {pe.get("id"):
                PeptideType.decoy if pe.get("isDecoy") == "true"
                else PeptideType.normal for pe in pep_evidences}

    def _extract_peptides(self, peptides) \
            -> Dict[str, Tuple[str, List[ModSite]]]:
        """
        Extracts the peptide information from the Peptide elements.

        Args:
            peptides: Iterable of Peptide XML elements.

        Returns:

        """
        results: Dict[str, Tuple[str, List[ModSite]]] = {}
        for peptide in peptides:
            seq = peptide.find(self._fix_tag("PeptideSequence")).text
            mods = self._extract_peptide_mods(
                peptide.findall(self._fix_tag("Modification")), seq)
            results[peptide.get("id")] = (seq, mods)
        return results

    def _extract_peptide_mods(self, mods, seq: str) -> List[ModSite]:
        """
        Extracts the peptide modifications from the Modification elements.

        Args:
            mods: Iterable of Modification XML elements.
            seq (str): The peptide amino acid sequence.

        Returns:
            The ModSites in a list.

        """
        mod_sites: List[ModSite] = []
        for mod_elem in mods:
            site: Union[str, int] = int(mod_elem.get("location"))
            if site == 0:
                site = "nterm"
            elif site == len(seq) + 1:
                site = "cterm"
            name = html.unescape(
                mod_elem.find(self._fix_tag("cvParam")).get("name"))
            mass = float(mod_elem.get("monoisotopicMassDelta",
                                      default=mod_elem.get("avgMassDelta")))
            mod_sites.append(ModSite(mass, site, name))
        return mod_sites

    def _extract_spec_idents(self, spec_results) \
            -> Dict[Tuple[str, Optional[str]], List[Ident]]:
        """
        Extracts the peptide identifications for all spectra.

        """
        idents: Dict[Tuple[str, Optional[str]], List[Ident]] = {}
        for spec in spec_results:
            idents[(self._parse_id(spec.get("spectrumID")), None)] = \
                self._extract_spec_peptides(
                    spec.findall(self._fix_tag("SpectrumIdentificationItem")))
        return idents

    def _extract_spec_peptides(self, ident_items) -> List[Ident]:
        """
        Extracts the peptide identifications for a spectrum.

        """
        return [
            Ident(e.get("peptide_ref"),
                  e.find(self._fix_tag("PeptideEvidenceRef"))
                  .get("peptideEvidence_ref"),
                  int(e.get("chargeState")),
                  int(e.get("rank")),
                  float(e.get("experimentalMassToCharge")),
                  float(e.get("calculatedMassToCharge")),
                  e.get("passThreshold") == "true",
                  {k: float(v) for k, v in self._get_cv_params(e).items()})
            for e in ident_items]

    def _parse_id(self, id_str: str) -> str:
        """
        Parses the ID string to extract the numeric elements.

        """
        return ".".join(re.findall(r"\w+=(\d+)", id_str))

    def _get_cv_params(self, element) -> Dict[str, str]:
        """
        Parses the cvParam elements under the given XML element to a dict.

        Args:
            element: An XML element.

        Returns:
            A dictionary mapping the cvParam name to value (float).

        """
        return {e.get("name"): e.get("value")
                for e in element.findall(self._fix_tag("cvParam"))}

    def _fix_tag(self, tag: str) -> str:
        """
        Prepends the namespace to an XML tag.

        Args:
            tag (str): The XML tag to modify.

        """
        return f"{{{self.namespace}}}{tag}"
