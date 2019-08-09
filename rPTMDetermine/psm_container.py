#! /usr/bin/env python3
"""
A module providing a container class for PSM objects.

"""
from __future__ import annotations  # Imported for lazy evaluation of types

import collections
import csv
import operator
import os
import sys
from typing import (Callable, Dict, Generic, List, Optional, overload,
                    Sequence, Set, Tuple, TypeVar)

import pandas as pd

from pepfrag import ModSite, Peptide

from .peptide_spectrum_match import PSM, SimilarityScore
from .readers import parse_mods


PSMType = TypeVar("PSMType", bound=PSM)


class PSMContainer(collections.UserList, Generic[PSMType]):  # pylint: disable=too-many-ancestors
    """
    A class to provide a customized iterable container of PSMs. The class
    builds upon the UserList class and extends the functionality.

    """
    def __init__(self, psms: Optional[List[PSMType]] = None):
        """
        Initialize the instance of the class.

        """
        super().__init__()
        # TODO
        # The generic type of self.data needs to be overridden, but this is
        # problematic due to https://github.com/python/mypy/issues/5846
        self.data = psms if psms is not None else []

    @overload
    def __getitem__(self, idx: int) -> PSMType: ...

    @overload
    def __getitem__(self, idx: slice) -> PSMContainer[PSMType]: ...

    def __getitem__(self, idx):
        res = self.data[idx]
        return PSMContainer(res) if isinstance(res, list) else res

    def clean_fragment_ions(self):
        """
        Removes the cached fragment ions for the PSMs.

        """
        for psm in self.data:
            psm.peptide.clean_fragment_ions()

    def get_by_seq(self, seq: str) -> PSMContainer[PSMType]:
        """
        Retrieves the PSMs with the given peptide sequence.

        Args:
            seq (str): The peptide sequence.

        Returns:
            PSMContainer

        """
        return PSMContainer([p for p in self.data if p.seq == seq])

    def get_by_id(self, data_id: str, spec_id: str) -> PSMContainer[PSMType]:
        """
        Retrieves the PSMs with the given identifiers.

        Args:
            data_id (str): The data set ID.
            spec_id (str): The spectrum ID within the data set.

        Returns:
            PSMContainer

        """
        return PSMContainer([p for p in self.data if p.data_id == data_id and
                             p.spec_id == spec_id])

    def filter_lda_prob(self, threshold: float = 0.99) \
            -> PSMContainer[PSMType]:
        """
        Filters the PSMs to those with an LDA probability exceeding the
        threshold value.

        Args:
            threshold (float): The threshold probability to exceed.

        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data if p.lda_prob >= threshold])

    def filter_lda_similarity(self, lda_threshold: float,
                              sim_threshold: float) -> PSMContainer[PSMType]:
        """
        Filters the PSMs to those with an LDA prob exceeding the threshold
        value and a maximum similarity score exceeding the similarity score
        threshold.

        Args:
            lda_threshold (float): The threshold LDA probability to exceed.
            sim_threshold (float): The similarity score threshold to exceed.

        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data if p.lda_prob >= lda_threshold and
             p.max_similarity >= sim_threshold])

    def filter_site_prob(self, threshold: float) -> PSMContainer[PSMType]:
        """
        Filters the PSMs to those without a site probability or with a site
        probability exceeding the threshold.

        Args:
            threshold (float): The site probability threshold.

        Returns:
            PSMContainer

        """
        return PSMContainer([
            p for p in self.data if p.site_prob is None or
            p.site_prob >= threshold])

    def ids_not_in(self, exclude_ids: Sequence[Tuple[str, str]])\
            -> PSMContainer[PSMType]:
        """
        Filters the PSMs to those whose (data_id, spec_id) pair is not in the
        exclude list provided.

        Args:
            exclude_ids (list): A list of (data_id, spec_id) tuples.

        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data
             if (p.data_id, p.spec_id) not in exclude_ids])

    def get_index(self, attributes: Sequence[str]) \
            -> Dict[Tuple[str, ...], List[int]]:
        """
        Builds a dictionary index mapping the values of the given attribute to
        a list of positions in the PSMContainer.

        Args:
            attribute (str): The PSM attribute on which to build an index.

        Returns:
            Index dictionary mapping values to positions.

        """
        index: Dict[Tuple[str, ...], List[int]] = collections.defaultdict(list)
        for idx, psm in enumerate(self.data):
            index[tuple([getattr(psm, a) for a in attributes])].append(idx)

        return index

    def get_best_psms(self) -> PSMContainer[PSMType]:
        """
        Extracts only the PSM with the highest LDA score for each spectrum
        matched by any number of peptides.

        Returns:
            PSMContainer of filtered PSMs.

        """
        index = self.get_index(("data_id", "spec_id"))

        return PSMContainer([max([self.data[i] for i in indices],
                                 key=operator.attrgetter("lda_score"))
                             for indices in index.values()])

    def get_unique_peptides(
            self, predicate: Optional[Callable[[PSM], bool]] = None)\
            -> Set[Tuple[str, Tuple[ModSite]]]:
        """
        Finds the unique peptides, by sequence and modifications.

        Args:
            predicate (func, optional): If not None, this function will be
                                        evaluated for each PSM and the peptide
                                        will only be returned if the return
                                        value of this function is True.

        Returns:
            Set of unique peptides as tuples of sequence and mods.

        """
        peptides: Set[Tuple[str, Tuple[ModSite]]] = set()
        for psm in self.data:
            if predicate is None or predicate(psm):
                # TODO: typing ignore due to
                # https://github.com/python/mypy/issues/5846
                peptides.add((psm.seq, tuple(psm.mods)))  # type: ignore
        return peptides

    def get_benchmark_peptides(self)\
            -> Set[Tuple[str, Tuple[ModSite]]]:
        """
        Finds the unique peptides, by sequence and modifications, which
        correspond to benchmark peptides.

        Returns:
            Set of unique benchmark peptides as tuples of sequence and mods.

        """
        return self.get_unique_peptides(predicate=lambda psm: psm.benchmark)

    def get_validated(
        self,
        lda_threshold: float,
        sim_threshold: float,
        site_prob: float) -> PSMContainer[PSMType]:
        """
        Extracts the validated identifications from the full set of PSMs.

        Args:
            lda_threshold (float): The LDA score threshold.
            sim_threshold (float): The similarity score threshold.
            site_prob (float): The minimum site probability for localization.

        Returns:
            A PSMContainer containing only validated PSMs.

        """
        return PSMContainer(
            [psm for psm in self.data if
             round(psm.lda_score, 2) >= lda_threshold and
             round(psm.max_similarity, 2) >= sim_threshold and
             (psm.site_prob is None or psm.site_prob >= site_prob)])

    def to_df(self, target_only: bool = False) -> pd.DataFrame:
        """
        Converts the psm features, including decoy, into a pandas dataframe,
        including a flag indicating whether the features correspond to a
        target or decoy peptide and the peptide sequence.

        Returns:
            pandas.DataFrame

        """
        rows = []
        for psm in self.data:
            trow = {"data_id": psm.data_id, "spec_id": psm.spec_id,
                    "seq": psm.seq, "target": True, "uid": psm.uid}
            for feature, value in psm.features:
                trow[feature] = value
            rows.append(trow)
            if not target_only and psm.decoy_id is not None:
                drow = {"data_id": "", "spec_id": "",
                        "seq": psm.decoy_id.seq, "target": False,
                        "uid": psm.uid}
                for feature, value in psm.decoy_id.features:
                    drow[feature] = value
                rows.append(drow)

        return pd.DataFrame(rows)


def read_csv(csv_file: str, ptmdb, spectra=None, sep: str = "\t")\
        -> PSMContainer:
    """
    Converts the contents of a CSV file to a PSMContainer. Each row of the
    file should contain the details of a single PSM. This function is
    designed to be used to read CSV files output by this program.

    Args:
        csv_file (str): The path to the CSV file.

    Returns:
        PSMContainer.

    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist - exiting")
        sys.exit(1)

    psms = []
    with open(csv_file, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        for row in reader:
            psm = PSM(
                row.get("DataID", row.get("Rawset", None)),
                row["SpectrumID"],
                Peptide(
                    row["Sequence"],
                    int(row["Charge"]),
                    parse_mods(row["Modifications"], ptmdb)
                )
            )
            psm.lda_score = float(row["rPTMDetermineScore"])
            psm.lda_prob = float(row["rPTMDetermineProb"])
            site_prob = row["SiteProbability"]
            psm.site_prob = (float(site_prob) if site_prob else None)

            if "SimilarityScore" in row:
                psm.similarity_scores = [SimilarityScore(
                    None, None, float(row["SimilarityScore"]))]

            if spectra is not None:
                psm.spectrum = spectra[psm.data_id][psm.spec_id]

            psms.append(psm)

    return PSMContainer(psms)
