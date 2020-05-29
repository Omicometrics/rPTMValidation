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
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    overload, Sequence, Set, Tuple, TypeVar, TYPE_CHECKING)

import numpy as np

from pepfrag import ModSite, Peptide

from . import (
    machinelearning
)
from .peptide_spectrum_match import PSM
from .readers import parse_mods, PTMDB, UnknownModificationException

if TYPE_CHECKING:
    # Optional dependencies are specified here to enable type checking
    # (see https://stackoverflow.com/questions/61384752)
    import pandas as pd


PSMType = TypeVar("PSMType", bound=PSM)


# TODO: use SQLAlchemy, backed by sqlite3, to create a PSM database, which
#       can be queried to enable quick lookups. Would use a "spectra" table
#       to independently store `Spectrum` objects, then join on the
#       (data_id, spec_id) with "psms" table
class PSMContainer(collections.UserList, Generic[PSMType]):  # pylint: disable=too-many-ancestors
    """
    A class to provide a customized iterable container of PSMs. The class
    builds upon the UserList class and extends the functionality.

    """
    def __init__(self, psms: Optional[Iterable[PSMType]] = None):
        """
        Initialize the instance of the class.

        """
        super().__init__()

        self.data: List[PSMType] = list(psms) if psms is not None else []

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

    def ids_not_in(self, exclude_ids: Iterable[Tuple[str, str]])\
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
            attributes: The PSM attributes on which to build an index.

        Returns:
            Index dictionary mapping values to positions.

        """
        index: Dict[Tuple[str, ...], List[int]] = collections.defaultdict(list)
        for idx, psm in enumerate(self.data):
            index[tuple([getattr(psm, a) for a in attributes])].append(idx)

        return index

    def get_best_psms(self, threshold: float) -> PSMContainer[PSMType]:
        """
        Extracts only the PSM with the greatest score sum for each spectrum
        matched by any number of peptides.

        Returns:
            PSMContainer of filtered PSMs.

        """
        index = self.get_index(("data_id", "spec_id"))

        best_psms: PSMContainer[PSMType] = PSMContainer()
        for indices in index.values():
            best = self.data[indices[0]]
            if len(indices) < 2:
                best_psms.append(best)
                continue

            best_score_sum = best.ml_scores.sum()
            best_passes = machinelearning.passes_consensus(
                best.ml_scores, threshold
            )
            for index in indices[1:]:
                psm = self.data[index]
                score_sum = psm.ml_scores.sum()
                current_passes = machinelearning.passes_consensus(
                    psm.ml_scores, threshold
                )
                if ((current_passes and not best_passes) or
                        (score_sum >= best_score_sum and current_passes) or
                        (score_sum >= best_score_sum and not current_passes and
                         not best_passes)):
                    best = psm
            best_psms.append(best)

        return best_psms

    def get_unique_peptides(
            self, predicate: Optional[Callable[[PSM], bool]] = None)\
            -> Set[Tuple[str, Tuple[ModSite, ...]]]:
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
        peptides: Set[Tuple[str, Tuple[ModSite, ...]]] = set()
        for psm in self.data:
            if predicate is None or predicate(psm):
                peptides.add((psm.seq, tuple(psm.mods)))

        return peptides

    def get_validated(
        self,
        score_threshold: float,
        use_consensus: bool = True
    ) -> PSMContainer[PSMType]:
        """
        Extracts the validated identifications from the full set of PSMs.

        Args:
            score_threshold: The classification score threshold.
            use_consensus: If True, strict score consensus is required. If
                           False, majority voting will be used instead.

        Returns:
            A PSMContainer containing only validated PSMs.

        """
        pass_function = (machinelearning.passes_consensus if use_consensus
                         else machinelearning.passes_majority)
        return PSMContainer([
            psm for psm in self.data if
            pass_function(psm.ml_scores, score_threshold) and
            psm.is_localized()
        ])

    def to_df(self) -> pd.DataFrame:
        """
        Converts the psm features into a pandas dataframe, including the peptide
        sequence.

        Returns:
            pandas.DataFrame

        """
        import pandas as pd

        rows = []
        for psm in self.data:
            trow: Dict[str, Any] = {
                "data_id": psm.data_id,
                "spec_id": psm.spec_id,
                "seq": psm.seq,
                "uid": psm.uid
            }
            for feature, value in psm.features:
                trow[feature] = value
            rows.append(trow)

        return pd.DataFrame(rows)

    def to_feature_array(
        self,
        features: Optional[List[str]] = None
    ) -> np.array:
        """
        Constructs an `np.array` using the features of the contained `PSM`s.

        Args:
            features:

        Returns:
            numpy array

        """
        return np.array([psm.features.to_list(features) for psm in self.data])

    @classmethod
    def from_csv(
        cls,
        csv_file: str,
        ptmdb: PTMDB,
        spectra=None,
        sep: str = ','
    ) -> PSMContainer[PSM]:
        """
        Converts the contents of a CSV file to a PSMContainer. Each row of the
        file should contain the details of a single PSM. This function is
        designed to be used to read CSV files output by this program.

        Args:
            csv_file (str): The path to the CSV file.
            ptmdb:
            spectra:
            sep:

        Returns:
            PSMContainer.

        """
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} does not exist - exiting")
            sys.exit(1)

        psms: PSMContainer[PSM] = PSMContainer()
        with open(csv_file, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            for row in reader:
                mods_str = row.get('Modifications', row.get('Mods'))
                if mods_str is None:
                    raise KeyError(
                        f'No Modifications/Mods found in {csv_file}'
                    )
                try:
                    mods = parse_mods(mods_str, ptmdb)
                except UnknownModificationException:
                    continue

                psm = PSM(
                    row['DataID'],
                    row['SpectrumID'],
                    Peptide(
                        row['Sequence'],
                        int(row['Charge']),
                        mods
                    )
                )

                if spectra is not None:
                    psm.spectrum = spectra[psm.data_id][psm.spec_id]

                psm.ml_scores = np.array(
                    list(map(float, row['Scores'].split(';')))
                )

                def get_float_or_none(key):
                    try:
                        return float(row[key])
                    except ValueError:
                        return None

                psm.site_score = get_float_or_none('SiteScore')
                psm.site_prob = get_float_or_none('SiteProbability')
                psm.site_diff_score = get_float_or_none('SiteDiffScore')

                psms.append(psm)

        return psms

    @classmethod
    def from_csv_v1(
            cls,
            csv_file: str,
            ptmdb: PTMDB,
            spectra=None,
            sep: str = ','
    ) -> PSMContainer[PSM]:
        """
        Converts the contents of a CSV file to a PSMContainer. Each row of the
        file should contain the details of a single PSM. This function is
        designed to be used to read CSV files output by this program.

        Args:
            csv_file (str): The path to the CSV file.
            ptmdb:
            spectra:
            sep:

        Returns:
            PSMContainer.

        """
        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} does not exist - exiting")
            sys.exit(1)

        psms: PSMContainer[PSM] = PSMContainer()
        with open(csv_file, newline="") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            for row in reader:
                mods_str = row.get('Modifications', row.get('Mods'))
                if mods_str is None:
                    raise KeyError(
                        f'No Modifications/Mods found in {csv_file}'
                    )
                try:
                    mods = parse_mods(mods_str, ptmdb)
                except UnknownModificationException:
                    continue

                psm = PSM(
                    row['DataID'],
                    row['SpectrumID'],
                    Peptide(
                        row['Sequence'],
                        int(row['Charge']),
                        mods
                    )
                )

                if spectra is not None:
                    psm.spectrum = spectra[psm.data_id][psm.spec_id]

                psms.append(psm)

        return psms
