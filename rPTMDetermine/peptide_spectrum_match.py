#! /usr/bin/env python3
"""
Module contains a class to define a Peptide Spectrum Match (PSM).

"""
import bisect
import collections
import dataclasses
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

from .constants import DEFAULT_FRAGMENT_IONS, FIXED_MASSES
from . import ionscore, mass_spectrum, proteolysis, utilities

from pepfrag import IonType, ModSite, Peptide


DecoyID = collections.namedtuple(
    "DecoyID", ["seq", "charge", "mods", "features"])

SimilarityScore = collections.namedtuple(
    "SimilarityScore", ["data_id", "spectrum_id", "score"])


@dataclasses.dataclass(init=False)
class Features:
    """
    A data class to represent the available features.

    """
    __slots__ = ("NumPeaks", "TotInt", "PepLen", "PepMass", "Charge",
                 "ErrPepMass", "IntModyb", "TotalIntMod", "FracIon",
                 "FracIonInt", "NumSeriesbm", "NumSeriesym", "NumIona",
                 "NumIonynl", "NumIonbnl", "FracIonIntb_c1", "FracIonIntb_c2",
                 "FracIonIntb_c3", "FracIonInty_c1", "FracIonInty_c2",
                 "FracIonInty_c3", "FracIon20pc", "NumIonb", "NumIony",
                 "FracIonInty", "FracIonIntb", "FracIonMod", "MatchScore",
                 "MatchScoreMod", "SeqTagm",)

    NumPeaks: float
    TotInt: float
    PepLen: float
    PepMass: float
    Charge: float
    ErrPepMass: float
    IntModyb: float
    TotalIntMod: float
    FracIon: float
    FracIonInt: float
    NumSeriesbm: float
    NumSeriesym: float
    NumIona: float
    NumIonynl: float
    NumIonbnl: float
    FracIonIntb_c1: float
    FracIonIntb_c2: float
    FracIonIntb_c3: float
    FracIonInty_c1: float
    FracIonInty_c2: float
    FracIonInty_c3: float
    FracIon20pc: float
    NumIonb: float
    NumIony: float
    FracIonInty: float
    FracIonIntb: float
    FracIonMod: float
    MatchScore: float
    MatchScoreMod: float
    SeqTagm: float

    def __init__(self):
        """
        Initialize the data class by defaulting all features to 0.0.

        """
        for feature in Features.__slots__:
            setattr(self, feature, 0.)

    def feature_names(self) -> Tuple[str, ...]:
        """
        Extracts the available feature names.

        Returns:
            Tuple of feature names.

        """
        return Features.__slots__

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        """
        Iterates through the features to retrieve the corresponding values.

        Returns:
            Iterator of (feature, value) tuples.

        """
        for feature in self.feature_names():
            yield feature, getattr(self, feature)


class PSM():
    """
    A class to represent a Peptide Spectrum Match, containing details of the
    peptide and the matched mass spectrum. The mass spectrum is a composed
    Spectrum object, while the peptide is a composed pepfrag.Peptide.

    """

    __slots__ = ("data_id", "spec_id", "peptide", "__spectrum", "decoy_id",
                 "benchmark", "similarity_scores", "features", "lda_score",
                 "lda_prob", "decoy_lda_score", "decoy_lda_prob", "site_prob",
                 "corrected", "target",)

    def __init__(self, data_id: Optional[str], spec_id: Optional[str],
                 peptide: Peptide, spectrum=None, target=True):
        """
        Initializes the PSM class using basic identifying information.

        Args:
            data_id (str): The ID of the source data file.
            spec_id (str): The ID associated with the matched spectrum.
            peptide (pepfrag.Peptide): The peptide matched to the spectrum.
            spectrum (Spectrum, optional): The Spectrum matched to the
                                           peptide. This is usually set later.

        """
        self.data_id = data_id
        self.spec_id = spec_id
        self.peptide = peptide

        # This can be set later before processing occurs
        self.__spectrum = spectrum

        # The decoy peptide matched to the spectrum
        self.decoy_id: Optional[DecoyID] = None

        # Whether the spectrum has been assigned a peptide considered to be
        # a benchmark peptide
        self.benchmark = False

        # The similarity scores to the (unmodified) spectra
        self.similarity_scores: List[SimilarityScore] = []

        # The PSM features
        self.features = Features()

        # The results of LDA validation
        self.lda_score: Optional[float] = None
        self.lda_prob: Optional[float] = None
        self.decoy_lda_score: Optional[float] = None
        self.decoy_lda_prob: Optional[float] = None

        # The localization site probability
        self.site_prob: Optional[float] = None

        # Whether the PSM has been corrected by some post-processing, e.g.
        # removal of deamidation
        self.corrected = False

        self.target = target

    @property
    def seq(self) -> str:
        """
        Returns the peptide sequence.

        """
        return self.peptide.seq

    @property
    def mods(self) -> List[ModSite]:
        """
        Returns the peptide modifications list.

        """
        return self.peptide.mods

    @mods.setter
    def mods(self, val: List[ModSite]):
        """
        Sets the peptide modifications list.

        """
        self.peptide.mods = val

    @property
    def charge(self) -> int:
        """
        Returns the peptide charge state.

        """
        return self.peptide.charge

    @property
    def spectrum(self) -> mass_spectrum.Spectrum:
        """
        Returns the composed mass_spectrum.Spectrum.

        """
        return self.__spectrum

    @spectrum.setter
    def spectrum(self, val):
        """
        Sets the composed mass_spectrum.Spectrum.

        """
        if val is not None and not isinstance(val, mass_spectrum.Spectrum):
            raise TypeError(
                "Setting PSM.spectrum requires a mass_spectrum.Spectrum")
        self.__spectrum = val

    @property
    def uid(self):
        """
        Returns the unique identifier for the PSM.

        """
        return f"{self.data_id}_{self.spec_id}_{self.seq}"

    @property
    def max_similarity(self):
        """
        Returns the maximum similarity score for the PSM.

        """
        return (0. if not self.similarity_scores
                else max(s[2] for s in self.similarity_scores))

    def __str__(self) -> str:
        """
        Implements the string conversion for the object.

        Returns:
            String representation.

        """
        out = {
            "data_id": self.data_id,
            "spec_id": self.spec_id,
            "peptide": self.peptide,
            "spectrum": self.spectrum,
            "decoy_id": self.decoy_id,
            "benchmark": self.benchmark,
            "similarity_scores": self.similarity_scores,
            "features": self.features,
            "lda_prob": self.lda_prob,
            "lda_score": self.lda_score,
            "decoy_lda_score": self.decoy_lda_score,
            "decoy_lda_prob": self.decoy_lda_prob,
            "site_prob": self.site_prob,
            "corrected": self.corrected,
            "target": self.target
        }
        return f"<{self.__class__.__name__} {out}>"

    def __repr__(self) -> str:
        """
        Implements the repr conversion for the object.

        Returns:
            Official string representation.

        """
        return f"<{self.__class__.__name__} {self.__dict__}>"

    def __hash__(self):
        """
        Implements the hash function for the object.

        """
        return hash((self.data_id, self.spec_id, self.peptide))

    def __eq__(self, other):
        """
        Implements the equality test for the object.

        """
        return (self.data_id, self.spec_id, self.peptide) == \
               (other.data_id, other.spec_id, other.peptide)

    def _check_spectrum_initialized(self):
        """
        Checks whether the spectrum attribute has been initialized to a
        Spectrum object.

        Raises:
            RuntimeError if self.spectrum is not assigned.

        """
        if self.spectrum is None:
            raise RuntimeError("PSM has not been assigned a Spectrum")

    def annotate_spectrum(
        self, tol: float = 0.2,
        ion_types: Optional[Dict[IonType, Dict[str, Any]]] = None)\
            -> Dict[str, mass_spectrum.Annotation]:
        """
        Annotates the mass spectrum using the theoretical ions of the peptide.

        Args:
            tol (float, optional): The annotation m/z tolerance.
            ion_types (dict, optional): The fragmentation configuration dict.

        Returns:
            Dictionary mapping ion labels to mass_spectrum.Annotation.

        """
        self._check_spectrum_initialized()

        if ion_types is None:
            ion_types = {
                IonType.precursor: {"neutral_losses": ["H2O", "NH3"]},
                IonType.imm: {},
                IonType.b: {"neutral_losses": ["H2O", "NH3"]},
                IonType.y: {"neutral_losses": ["H2O", "NH3"]},
                IonType.a: {"neutral_losses": []}
            }

        # Get the theoretical ions for the peptide
        ions = self.peptide.fragment(
            ion_types=DEFAULT_FRAGMENT_IONS if ion_types is None
            else ion_types)

        return self.spectrum.annotate(ions, tol=tol)

    def denoise_spectrum(self, tol: float = 0.2)\
            -> Tuple[Dict[str, Tuple[int, int]], mass_spectrum.Spectrum]:
        """
        Adaptively denoises the mass spectrum.

        Returns:
            A dictionary mapping ion labels to peaks.

        """
        self._check_spectrum_initialized()

        # The spectrum annotations
        anns = self.annotate_spectrum(tol=tol)
        ann_peak_nums = {an.peak_num for an in anns.values()}
        denoised_peaks, denoised_spec = self.spectrum.denoise(
            [idx in ann_peak_nums for idx in range(len(self.spectrum))])

        denoised_peaks.sort()

        ion_anns = {l: (bisect.bisect_left(denoised_peaks, a.peak_num),
                        a.ion_pos)
                    for l, a in anns.items() if a.peak_num in denoised_peaks}

        return ion_anns, denoised_spec

    def extract_features(self, target_mod: Optional[str],
                         proteolyzer: proteolysis.Proteolyzer,
                         tol: float = 0.2) -> Features:
        """
        Extracts possible machine learning features from the peptide spectrum
        match.

        Args:
            target_mod (str): The modification type under validation. If this,
                              is None, i.e. for unmodified analogues, then
                              some modification-based features will not be
                              calculated.
            proteolyzer (proteolysis.Proteolyzer): The enzymatic proteolyzer
                                                   for calculating the number
                                                   of missed cleavages.

        Returns:
            dictionary of calculated features

        """
        self._check_spectrum_initialized()

        ions, denoised_spectrum = self.denoise_spectrum()
        self.spectrum.normalize()
        denoised_spectrum.normalize()
        self._calculate_prop_features(denoised_spectrum)
        if ions:
            self._calculate_ion_features(ions,
                                         denoised_spectrum,
                                         target_mod,
                                         tol)

        # Use the proteolyzer to determine the number of missed cleavages
        # self.features['n_missed_cleavages'] =\
            # proteolyzer.count_missed_cleavages(self.seq)

        return self.features

    def _calculate_prop_features(
            self, denoised_spectrum: mass_spectrum.Spectrum) -> Features:
        """
        Calculate statistics of the denoised mass spectrum
        and the matched peptide
        """
        intensities = (denoised_spectrum.intensity)
        # Number of peaks
        npeak = denoised_spectrum.intensity.size
        # The number of peaks, log transformed
        self.features.NumPeaks = np.log(npeak)
        totalint = intensities.sum()
        self.features.TotInt = totalint

        # The length of the peptide
        self.features.PepLen = len(self.seq)

        # Experimental m/z
        pms = (denoised_spectrum.prec_mz - FIXED_MASSES["H"]) * self.charge
        # Peptide related features
        self.features.PepMass = np.log(pms)
        self.features.Charge = self.charge
        self.features.ErrPepMass = abs(self.peptide.mass - pms)

        return self.features

    def _calculate_ion_features(self, ions: dict,
                                denoised_spectrum: mass_spectrum.Spectrum,
                                target_mod: Optional[str],
                                tol: float) -> Features:
        """
        Calculates potential machine learning features from the peptide
        spectrum match.

        Args:
            ions (dict): The theoretical ion peak annotations.
            denoised_spectrum (Spectrum): The denoised mass spectrum.
            target_mod (str): The target modification type. If this, is None,
                              i.e. for unmodified analogues, then some
                              modification-based features will not be
                              calculated.
            tol (float): The mass tolerance level to apply.

        """
        # The length of the peptide
        pep_len = len(self.seq)
        len_normalizer = np.sqrt(pep_len)
        intensities = (denoised_spectrum.intensity
                       / denoised_spectrum.max_intensity())

        # Number of peaks
        npeak = denoised_spectrum.intensity.size
        # The number of peaks, log transformed
        totalint = intensities.sum()

        mod_ion_start: Dict[str, int] = {}
        if target_mod is not None:
            # The position from which b-/y-ions will contain the modified
            # residue
            mod_ion_start = {"b": min(ms.site for ms in self.mods
                                      if ms.mod == target_mod),
                             "a": min(ms.site for ms in self.mods
                                      if ms.mod == target_mod),
                             "y": min(pep_len - ms.site + 1
                                      for ms in self.mods
                                      if ms.mod == target_mod)}

            # The sum of the modified ion intensities
            self.features.IntModyb = \
                sum(intensities[ions[l][0]] for l in ions.keys()
                    if (l[0] == "y" and "-" not in l and
                        ions[l][1] >= mod_ion_start["y"])
                    or (l[0] == "b" and "-" not in l and
                        ions[l][1] >= mod_ion_start["b"]))

        # The intensity of the base peak
        max_int = denoised_spectrum.max_intensity()
        # The peaks with intensity >= 20% of the base peak intensity
        peaks_20, = np.where((denoised_spectrum.mz >= 300)
                             & (denoised_spectrum.intensity >= max_int * 0.2))

        # indices of different types of ions
        charged_ion_indices: Dict[str, Dict[int, List[str]]] = \
            collections.defaultdict(lambda: collections.defaultdict(list))
        ion_indices: Dict[str, List[str]] = collections.defaultdict(list)
        seq_ions: List[str] = []
        seq_mod_ions: List[str] = []
        # get the indices
        for ion, (peakj, _) in ions.items():
            iontag = set(ion[:2]) & set("yba")
            if not iontag:
                continue
            c = 1 if ion.endswith("[+]") else int(ion[ion.index("+]") - 1])
            if ion[0] == "y":
                if c <= 3:
                    charged_ion_indices["y"][c].append(peakj)
                seq_ions.append(ion)
            elif ion[0] == "b":
                if c <= 3:
                    charged_ion_indices["b"][c].append(peakj)
                seq_ions.append(ion)
            elif ion[0] == "a":
                ion_indices["a"].append(peakj)
            elif "-" in ion and ion[1] == "y":
                ion_indices["ynl"].append(peakj)
            elif "-" in ion and ion[1] == "b":
                ion_indices["bnl"].append(peakj)

            if peakj >= mod_ion_start[list(iontag)[0]]:
                seq_mod_ions.append(peakj)

        # The total sum of the modified ion intensities
        self.features.TotalIntMod = intensities[seq_mod_ions].sum()

        # The peaks annotated by theoretical ions
        ann_peaks = np.array(list({v for v, _ in ions.values()}))

        # The number of annotated peaks divided by the total number of peaks
        self.features.FracIon = ann_peaks.size / npeak
        # The fraction of annotated peak intensities divided by
        # The total intensity of the spectrum
        self.features.FracIonInt = intensities[ann_peaks].sum() / totalint

        # Sequence coverage
        n_anns = self._calculate_sequence_coverage(target_mod, seq_ions,
                                                   mod_ion_start)

        # The fraction of b-ions annotated by theoretical ions
        self.features.NumSeriesbm /= len_normalizer
        # The fraction of y-ions annotated by theoretical ions
        self.features.NumSeriesym /= len_normalizer
        # The number of a ions
        self.features.NumIona = len(ion_indices["a"])
        # The number of neutral losses of a ions
        self.features.NumIonynl = len(ion_indices["ynl"])
        # The number of neutral losses of b ions
        self.features.NumIonbnl = len(ion_indices["bnl"])
        # The fraction of total intensities of different ion series
        # NOTE: The charge state of each fragment ion series is up to 3
        bion_ix: List[str] = []
        yion_ix: List[str] = []
        for c in [1, 2, 3]:
            # b series
            try:
                ix = charged_ion_indices["b"][c]
            except KeyError:
                continue
            bion_ix += ix
            setattr(self.features, f"FracIonIntb_c{c}",
                    intensities[ix].sum() / len_normalizer)

            # y series
            try:
                ix = charged_ion_indices["y"][c]
            except KeyError:
                continue
            yion_ix += ix
            setattr(self.features, f"FracIonInty_c{c}",
                    intensities[ix].sum() / len_normalizer)

        # The fraction of peaks with intensities greater than 20% of the base
        # peak annotated by the theoretical ions
        ions_20 = set(bion_ix + yion_ix) & set(peaks_20)
        if peaks_20.size > 0:
            self.features.FracIon20pc = len(ions_20) / peaks_20.size

        # The number of b ions
        self.features.NumIonb = len(bion_ix) / len_normalizer
        # The number of y ions
        self.features.NumIony = len(yion_ix) / len_normalizer
        # The fraction of total intensities of y ions
        self.features.FracIonInty = intensities[yion_ix].sum() / totalint
        # The fraction of total intensities of b ions
        self.features.FracIonIntb = intensities[bion_ix].sum() / totalint
        # The coverage of modified y and b ions
        self.features.FracIonMod = n_anns["mod"] / pep_len

        # Ion score
        self._calculate_ion_scores(denoised_spectrum, n_anns, target_mod, tol)

        return self.features

    def _calculate_ion_scores(self, denoised_spectrum: mass_spectrum.Spectrum,
                              n_anns: Dict[str, int],
                              target_mod: Optional[str],
                              tol: float):
        """
        Calculates the ion score features.

        """
        mzs = denoised_spectrum.mz
        mzrange = mzs[-1] - mzs[0]
        self.features.MatchScore = ionscore.ionscore(
            len(self.seq), len(mzs), n_anns["all"], mzrange, tol)
        self.features.MatchScore *= np.sqrt(len(self.seq))
        if target_mod is not None:
            self.features.MatchScoreMod = ionscore.ionscore(
                len(self.seq), len(mzs), n_anns["mod"], mzrange, tol)
            self.features.MatchScoreMod *= np.sqrt(len(self.seq))

    def _calculate_sequence_coverage(self, target_mod: Optional[str],
                                     seq_ions: Iterable[str],
                                     mod_ion_start: Dict[str, int]) \
            -> Dict[str, int]:
        """
        Calculates features related to the sequence coverage by the ion
        annotations.

        Args:
        """
        # The maximum number of fragments annotated by theoretical ions
        # across the charge states. mod is the maximum number of fragments
        # containing the modified residue
        n_anns: Dict[str, int] = {"all": 0, "mod": 0}

        # The longest consecutive ion sequence found among charge states
        max_ion_seq_len = 0
        # Across the charge states, the maximum number of each ion type
        # annotated
        max_ion_counts = {"b": -1, "y": -1}

        for _charge in range(self.charge):
            c_str = '[+]' if _charge == 0 else f'[{_charge + 1}+]'
            # The number of annotated ions
            n_ions = {"all": 0, "mod": 0}

            for ion_type in ["y", "b"]:
                # A list of b-/y-ion numbers (e.g. 2 for b2[+])
                ion_nums = sorted(
                    [int(l.split('[')[0][1:])
                     for l in seq_ions if l[0] == ion_type and c_str in l])

                if target_mod is not None:
                    # The number of ions of ion_type containing the
                    # modification
                    n_ions["mod"] += len(ion_nums) - \
                        bisect.bisect_left(ion_nums, mod_ion_start[ion_type])

                # Increment the number of ions annotated for the current charge
                n_ions["all"] += len(ion_nums)

                # Update the max number of annotated ions if appropriate
                if len(ion_nums) > max_ion_counts[ion_type]:
                    max_ion_counts[ion_type] = len(ion_nums)

                ion_seq_len, _ = utilities.longest_sequence(ion_nums)
                if ion_seq_len > max_ion_seq_len:
                    max_ion_seq_len = ion_seq_len

            if n_ions["all"] > n_anns["all"]:
                n_anns["all"] = n_ions["all"]

            if target_mod is not None and n_ions["mod"] > n_anns["mod"]:
                n_anns["mod"] = n_ions["mod"]

        self.features.NumSeriesbm = max_ion_counts["b"]
        self.features.NumSeriesym = max_ion_counts["y"]

        # The longest sequence tag found divided by the peptide length
        self.features.SeqTagm = max_ion_seq_len / float(len(self.seq))

        return n_anns


class UnmodPSM(PSM):
    """
    A simple subclass of PSM to represent an unmodified PSM. This includes
    an identifier which makes the unmodified PSM to its modified counterpart.

    """
    def __init__(self, mod_psm_uid: str, *args, **kwargs) -> None:
        """
        Initialize the UnmodPSM object by storing the modified PSM ID and
        passing the remaining arguments to the base class initializer.

        Args:
            mod_psm_uid (str): The modified counterpart identifier.

        """
        self.mod_psm_uid = mod_psm_uid
        super().__init__(*args, **kwargs)

    def __hash__(self):
        """
        Implements the hash function for the object.

        """
        return hash((self.mod_psm_uid, self.data_id, self.spec_id,
                     self.peptide))

    def __eq__(self, other):
        """
        Implements the equality test for the object.

        """
        return (self.mod_psm_uid, self.data_id, self.spec_id,
                self.peptide) == \
               (other.mod_psm_uid, other.data_id, other.spec_id,
                other.peptide)


def unique_unmod_psms(psms: List[UnmodPSM]) -> List[UnmodPSM]:
    """
    Generates a list of the unique UnmodPSMs, in terms of their underlying
    PSM only. If the mod_psm_uid is to be included in the uniqueness
    categorization, then set can be used directly.

    Args:
        psms (list): The unmodified PSMs.

    Returns:
        The list of unique unmodified PSMs.

    """
    uids: Set[str] = set()
    unique: List[UnmodPSM] = []
    for psm in psms:
        if psm.uid in uids:
            continue
        uids.add(psm.uid)
        unique.append(psm)
    return unique
