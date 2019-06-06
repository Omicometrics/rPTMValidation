#! /usr/bin/env python3
"""
Module contains a class to define a Peptide Spectrum Match (PSM).

"""
import bisect
import collections
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from pepfrag import FIXED_MASSES, IonType, ModSite, Peptide

from .constants import DEFAULT_FRAGMENT_IONS
from . import ionscore
from . import mass_spectrum
from . import proteolysis
from . import utilities


DecoyID = \
    collections.namedtuple("DecoyID", ["seq", "charge", "mods", "features"])

SimilarityScore = \
    collections.namedtuple("SimilarityScore",
                           ["data_id", "spectrum_id", "score"])


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
        self.features: Dict[str, float] = {}

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
                         proteolyzer: proteolysis.Proteolyzer) \
            -> Dict[str, float]:
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
        self._calculate_features(ions, denoised_spectrum, target_mod, 0.2)

        # Use the proteolyzer to determine the number of missed cleavages
        self.features['n_missed_cleavages'] =\
            proteolyzer.count_missed_cleavages(self.seq)

        return self.features

    def _calculate_features(self, ions: dict,
                            denoised_spectrum: mass_spectrum.Spectrum,
                            target_mod: Optional[str], tol: float)\
            -> Dict[str, float]:
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
        self.features["PepLen"] = pep_len

        self.features["PepMass"] = self.peptide.mass
        self.features["Charge"] = self.charge
        self.features["ErrPepMass"] = abs(
            self.features["PepMass"] - denoised_spectrum.prec_mz * self.charge
            + self.charge * FIXED_MASSES["H"])

        # Intensities from the spectrum
        intensities = list(denoised_spectrum.intensity)

        mod_ion_start: Dict[str, int] = {}
        if target_mod is not None:
            # The position from which b-/y-ions will contain the modified
            # residue
            mod_ion_start = {'b': min(ms.site for ms in self.mods
                                      if ms.mod == target_mod),
                             'y': min(pep_len - ms.site + 1
                                      for ms in self.mods
                                      if ms.mod == target_mod)}

            # The sum of the modified ion intensities
            self.features["TotalIntMod"] = \
                sum(intensities[ions[l][0]] for l in ions.keys()
                    if (l[0] == 'y' and '-' not in l and
                        ions[l][1] >= mod_ion_start['y'])
                    or (l[0] == 'b' and '-' not in l and
                        ions[l][1] >= mod_ion_start['b']))

        # The regular b-/y-ions annotated for the PSM
        seq_ions = [l for l in ions.keys() if l[0] in 'yb' and '-' not in l]

        # The peaks annotated by theoretical ions
        ann_peaks = {v[0] for v in ions.values()}

        # The number of annotated peaks divided by the total number of peaks
        self.features["FracIon"] = len(ann_peaks) /\
            float(len(denoised_spectrum))
        self.features["FracIonInt"] =\
            sum(intensities[idx] for idx in ann_peaks) / sum(intensities)

        # The intensity of the base peak
        max_int = denoised_spectrum.max_intensity()

        # The peaks with intensity >= 20% of the base peak intensity
        peaks_20 = {ii for ii, peak in enumerate(denoised_spectrum)
                    if peak[1] >= max_int * 0.2 and peak[0] >= 300}

        # The fraction of peaks with intensities greater than 20% of the base
        # peak annotated by the theoretical ions
        ions_20 = {ions[l][0] for l in seq_ions if ions[l][0] in peaks_20}
        self.features["FracIon20%"] = \
            (len(ions_20) / float(len(peaks_20)) if peaks_20 else 0)

        # Sequence coverage
        n_anns = self._calculate_sequence_coverage(target_mod, seq_ions,
                                                   mod_ion_start)

        # The fraction of b-ions annotated by theoretical ions
        self.features["NumIonb2L"] = self.features["NumIonb"] / float(pep_len)
        # The fraction of y-ions annotated by theoretical ions
        self.features["NumIony2L"] = self.features["NumIony"] / float(pep_len)

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
        self.features["MatchScore"] = ionscore.ionscore(
            len(self.seq), len(mzs), n_anns["all"], mzrange, tol)
        if target_mod is not None:
            self.features["MatchScoreMod"] = ionscore.ionscore(
                len(self.seq), len(mzs), n_anns["mod"], mzrange, tol)

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
        max_ion_counts = {'b': -1, 'y': -1}

        for _charge in range(self.charge):
            c_str = '[+]' if _charge == 0 else f'[{_charge + 1}+]'
            # The number of annotated ions
            n_ions = {"all": 0, "mod": 0}

            for ion_type in ['y', 'b']:
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

        self.features["NumIonb"] = max_ion_counts['b']
        self.features["NumIony"] = max_ion_counts['y']

        # The longest sequence tag found divided by the peptide length
        self.features["SeqTagm"] = max_ion_seq_len / float(len(self.seq))

        return n_anns


class UnmodPSM(PSM):
    """
    A simple subclass of PSM to represent an unmodified PSM. This includes
    an identifier which makes the unmodified PSM to its modified counterpart.

    """

    __slots__ = ("mod_psm_uid",)

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
