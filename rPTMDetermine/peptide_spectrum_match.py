#! /usr/bin/env python3
"""
Module contains a class to define a Peptide Spectrum Match (PSM).

"""
import bisect
import collections
import itertools
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .constants import DEFAULT_FRAGMENT_IONS, FIXED_MASSES
from .features import Features
from .base import SimilarityScore, Annotation, LocInfo, ModLocates
import rPTMDetermine.ionscore as ionscore
from . import mass_spectrum
from . import utilities

from pepfrag import IonType, ModSite, Peptide


class SpectrumNotFoundError(Exception):
    pass


class PSM:
    """
    A class to represent a Peptide Spectrum Match, containing details of the
    peptide and the matched mass spectrum. The mass spectrum is a composed
    Spectrum object, while the peptide is a composed pepfrag.Peptide.

    """

    __slots__ = (
        "data_id",
        "spec_id",
        "peptide",
        "_spectrum",
        "features",
        "annotations",
        "ml_scores",
        "validation_score",
        "similarity_scores",
        "target",
        "_localizations",
        "mod_locates"
    )

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
        self.target = target

        # This can be set later before processing occurs
        self.spectrum: mass_spectrum.Spectrum = spectrum

        # The PSM features
        self.features: Features = Features()

        # annotations once the spectrum is provided
        self.annotations: Optional[Annotation] = None

        # similarity scores
        self.similarity_scores: List[SimilarityScore] = []

        # The results of validation
        self.ml_scores: Optional[np.ndarray] = None
        self.validation_score: Optional[float] = None

        # Localization attributes
        self._localizations: Optional[Dict[str, LocInfo]] = None
        self.mod_locates: Optional[List[ModLocates]] = None

    @property
    def seq(self) -> str:
        """
        Returns the peptide sequence.

        """
        return self.peptide.seq

    @seq.setter
    def seq(self, val: str):
        """
        Sets the peptide sequence.

        """
        self.peptide.seq = val

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
        return self._spectrum

    @spectrum.setter
    def spectrum(self, val):
        """
        Sets the composed mass_spectrum.Spectrum.

        """
        if val is not None and not isinstance(val, mass_spectrum.Spectrum):
            raise TypeError(
                "Setting PSM.spectrum requires a mass_spectrum.Spectrum"
            )
        self._spectrum = val

    @property
    def uid(self):
        """
        Returns the unique identifier for the PSM.

        """
        return f"{self.data_id}@{self.spec_id}@{self.seq}"

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
            "features": self.features,
            "annotations": self.annotations,
            "ml_scores": self.ml_scores,
            "validation_score": self.validation_score,
            "localizations": self._localizations,
            "mod_locates": self.mod_locates,
            "similarity_scores": self.similarity_scores,
            "target": self.target
        }
        return f"<{self.__class__.__name__} {out}>"

    def __repr__(self) -> str:
        """
        Implements the repr conversion for the object.

        Returns:
            Official string representation.

        """
        out = {s: getattr(self, s) for s in self.__class__.__slots__}
        return f"<{self.__class__.__name__} {out}>"

    def __hash__(self):
        """
        Implements the hash function for the object.

        """
        return hash((self.data_id, self.spec_id, self.peptide))

    def __eq__(self, other):
        """
        Implements the equality test for the object.

        """
        if not isinstance(other, PSM):
            return NotImplemented
        return (self.data_id, self.spec_id, self.peptide) == \
               (other.data_id, other.spec_id, other.peptide)

    def _check_spectrum_initialized(self):
        """
        Checks whether the spectrum attribute has been initialized to a
        Spectrum object.

        Raises:
            SpectrumNotFoundError if self.spectrum is not assigned.

        """
        if self.spectrum is None:
            raise SpectrumNotFoundError(
                f"PSM ({self.uid}) has not been assigned a Spectrum"
            )

    @property
    def max_similarity(self) -> float:
        """
        Returns the maximum similarity score for the PSM.

        """
        return (0. if not self.similarity_scores
                else max(s[2] for s in self.similarity_scores))

    def annotate_spectrum(self, tol: float = 0.2, denoise=True,
                          ion_types: Optional[Dict[int, List[str]]] = None):
        """
        Annotates the mass spectrum using the theoretical ions of the peptide.

        Args:
            tol (float, optional): The annotation m/z tolerance.
            ion_types (dict, optional): The fragmentation configuration dict.
            denoise (bool, optional): Whether denoising is performed.

        Returns:
            List of Annotations.

        """
        # denoise mass spectrum
        if denoise:
            _, denoised_spec = self.denoise_spectrum(tol)
            # TODO: make the denoising optional.
        else:
            annotes = self._annotate_spectrum(tol, ion_types)

        frag_annotations = []
        for ion, anno in annotes.items():
            frag_annotations.append(Annotation(
                ion=ion,
                mz=self.spectrum.mz[anno.peak_num],
                peak_intensity=self.spectrum.intensity[anno.peak_num],
                mz_diff=anno.mass_diff
            ))

        self.annotations = frag_annotations

    def _annotate_spectrum(self, tol: float = 0.2,
                           ion_types: Optional[Dict[int, List[str]]] = None)\
            -> Dict[str, mass_spectrum.Annotation]:
        """ Annotate mass spectrum. """
        self._check_spectrum_initialized()

        if ion_types is None:
            ion_types = {
                IonType.precursor: ["H2O", "NH3"],
                IonType.imm: [],
                IonType.b: ["H2O", "NH3"],
                IonType.y: ["H2O", "NH3"],
                IonType.a: []
            }

        # Get the theoretical ions for the peptide
        ions = self.peptide.fragment(
            ion_types=DEFAULT_FRAGMENT_IONS if ion_types is None
            else ion_types)
        self.peptide.clean_fragment_ions()

        # annotate mass spectrum
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
        anns = self._annotate_spectrum(tol=tol)
        ann_peak_nums = {an.peak_num for an in anns.values()}
        denoised_peaks, denoised_spec = self.spectrum.denoise(
            [idx in ann_peak_nums for idx in range(len(self.spectrum))])

        denoised_peaks.sort()

        ion_anns = {
            lab: (bisect.bisect_left(denoised_peaks, a.peak_num), a.ion_pos)
            for lab, a in anns.items() if a.peak_num in denoised_peaks
        }

        return ion_anns, denoised_spec

    def is_localized(self) -> Dict[str, bool]:
        """
        Determines whether the PSM has been successfully localized.

        Note that this will also be `True` in the case where `site_diff_score`
        is `None`.

        Returns:
            Dictionary indicating localization status of a modification.

        """
        if self.mod_locates is None:
            raise ValueError("Didn't perform localization.")
        return {mloc.modification: mloc.is_localized
                for mloc in self.mod_locates}

    #######################
    # Feature Calculation #
    #######################

    def extract_features(
            self,
            tol: float = 0.2,
            required_features: Optional[Sequence[str]] = None
    ) -> Features:
        """
        Extracts possible machine learning features from the peptide spectrum
        match.

        Args:
            tol: Mass tolerance level.
            required_features: The features which should be set to 0. if their
                               values are not otherwise calculated.

        Returns:
            Features.

        """
        self._check_spectrum_initialized()

        ions, denoised_spec = self.denoise_spectrum()
        self.spectrum.normalize()
        denoised_spec.normalize()
        self._calculate_prop_features(denoised_spec)
        if ions:
            self._calculate_ion_features(self.seq, ions, denoised_spec, tol)

        if required_features is None:
            required_features = self.features.all_feature_names()
        for feature in required_features:
            if self.features.get(feature) is None:
                self.features.set(feature, 0.)

        return self.features

    def _calculate_prop_features(self, denoised_spectrum) -> Features:
        """
        Calculate statistics of the denoised mass spectrum
        and the matched peptide
        """
        intensities = denoised_spectrum.intensity
        # Number of peaks
        npeak = denoised_spectrum.intensity.size
        # The number of peaks, log transformed
        self.features.NumPeaks = np.log(npeak)
        totalint = intensities.sum()
        self.features.TotInt = totalint

        # Experimental m/z
        pms = (denoised_spectrum.prec_mz - FIXED_MASSES["H"]) * self.charge
        # Peptide related features
        self.features.PepMass = np.log(pms)
        self.features.Charge = self.charge

        return self.features

    def _calculate_ion_features(self, seq, ions, denoised_spectrum, tol):
        """
        Calculates potential machine learning features from the peptide
        spectrum match.

        Args:
            seq (str): peptide sequence
            ions (dict): The theoretical ion peak annotations.
            denoised_spectrum (Spectrum): The denoised mass spectrum.
            tol (float): The mass tolerance level to apply.

        """
        # The length of the peptide
        pep_len = len(seq)
        len_normalizer = np.sqrt(pep_len)
        intensities = (denoised_spectrum.intensity
                       / denoised_spectrum.max_intensity())

        # Number of peaks
        npeak = denoised_spectrum.intensity.size
        # The number of peaks, log transformed
        totalint = intensities.sum()

        # The intensity of the base peak
        max_int = denoised_spectrum.max_intensity()
        # The peaks with intensity >= 20% of the base peak intensity
        peaks_20, = np.where((denoised_spectrum.mz >= 300)
                             & (denoised_spectrum.intensity >= max_int * 0.2))

        # indices of different types of ions
        ion_indices: Dict[str, Set[str]] = collections.defaultdict(set)
        # each tuple: ion name, ion type, charge, peak index
        seq_ions: List[Tuple] = []
        ion_charges: List[int] = []
        # for getting complementary ions
        ion_seq_indices: Dict[int, Dict[str, set]] = collections.defaultdict(
            lambda: collections.defaultdict(set)
        )
        # get the indices
        for ion, (peakj, ionk) in ions.items():
            if ion.endswith("+]"):
                c = 1 if ion.endswith("[+]") else int(ion[ion.index("+]") - 1])
            else:
                c = 1
            ion_charges.append(c)
            iontag = set(ion[:2]) & set("yba")
            if not iontag or c > 2:
                continue

            if ion[0] in "yb":
                seq_ions.append((ion, ion[0], c, peakj))
                ion_seq_indices[c][ion[0]].add((peakj, ionk))
            elif ion[0] == "a":
                ion_indices["a"].add(peakj)
            elif "-" in ion and ion[1] == "y":
                ion_indices["ynl"].add(peakj)
            elif "-" in ion and ion[1] == "b":
                ion_indices["bnl"].add(peakj)

        # The peaks annotated by theoretical ions
        ann_peaks = np.unique(np.array(
            [v for c, (v, _) in zip(ion_charges, ions.values()) if c <= 2],
            dtype=int))

        # The number of annotated peaks divided by the total number of peaks
        self.features.FracIon = ann_peaks.size / npeak
        # The fraction of annotated peak intensities divided by
        # The total intensity of the spectrum
        self.features.FracIonInt = intensities[ann_peaks].sum() / totalint
        # The number of a ions
        self.features.NumIona = len(ion_indices["a"])
        # The number of neutral losses of a ions
        self.features.NumIonynl = len(ion_indices["ynl"])
        # The number of neutral losses of b ions
        self.features.NumIonbnl = len(ion_indices["bnl"])
        # The fraction of total intensities of different ion series
        # NOTE: The charge state of each fragment ion series is up to 2
        bion_ix: Set[int] = set()
        yion_ix: Set[int] = set()
        ncomp: int = 0
        idx_comp: List[int] = []
        ncomp_c1: int = 0
        idx_comp_c1: List[int] = []
        for c in [1, 2]:
            # b series
            ix = [j for _, t, cj, j in seq_ions if cj == c and t == "b"]
            bion_ix.update(ix)
            self.features.set(f"FracIonIntb_c{c}",
                              intensities[ix].sum() / len_normalizer)

            # y series
            ix = [j for _, t, cj, j in seq_ions if cj == c and t == "y"]
            yion_ix.update(ix)
            self.features.set(f"FracIonInty_c{c}",
                              intensities[ix].sum() / len_normalizer)

            # complementary ions
            for (k0, yk), (k1, bk) in itertools.product(
                    ion_seq_indices[c]["y"], ion_seq_indices[c]["b"]
            ):
                if yk + bk == pep_len:
                    ncomp += 1
                    idx_comp += [k0, k1]
                    if c == 1:
                        idx_comp_c1 += [k0, k1]
                        ncomp_c1 += 1

        # The fraction of peaks with intensities greater than 20% of the base
        # peak annotated by the theoretical ions
        ions_20 = bion_ix.union(yion_ix) & set(peaks_20)
        self.features.FracIon20pc = (
            len(ions_20) / peaks_20.size if peaks_20.size > 0 else 0)

        # The number of b ions
        self.features.NumIonb = len(bion_ix) / len_normalizer
        # The number of y ions
        self.features.NumIony = len(yion_ix) / len_normalizer
        # The fraction of total intensities of y ions
        self.features.FracIonInty = intensities[list(yion_ix)].sum() / totalint
        # The fraction of total intensities of b ions
        self.features.FracIonIntb = intensities[list(bion_ix)].sum() / totalint
        # The fraction of number of complementary ions
        self.features.FracNumComp = ncomp / len_normalizer
        # The fraction of total intensities of complementary ions
        self.features.FracIntComp = intensities[idx_comp].sum() / totalint
        # The fraction of number of complementary ions, singly charged
        self.features.FracNumComp_c1 = ncomp_c1 / len_normalizer
        # The fraction of total intensities of complementary ions,
        # singly charged
        self.features.FracIntComp_c1 = (intensities[idx_comp_c1].sum()
                                        / totalint)
        # The fraction of y and b ions annotated to all theoretical
        # y and b ions
        self.features.Numyb2Theo = ((len(yion_ix) + len(bion_ix))
                                    / (2 * (pep_len - 1)))

        # Sequence coverage
        n_anns = self._calculate_sequence_coverage(seq_ions)
        # The longest sequence tag found divided by the peptide length
        self.features.SeqTagm /= len_normalizer
        # The fraction of b-ions annotated by theoretical ions
        self.features.NumSeriesbm /= len_normalizer
        # The fraction of y-ions annotated by theoretical ions
        self.features.NumSeriesym /= len_normalizer
        # Ion score
        self._calculate_ion_scores(denoised_spectrum, n_anns, pep_len, tol)

        return self

    def _calculate_ion_scores(self, denoised_spectrum, n_anns, pep_len, tol):
        """
        Calculates the ion score features.

        """
        mzs = denoised_spectrum.mz
        mzrange = mzs[-1] - mzs[0]
        self.features.MatchScore = ionscore.ionscore(
            pep_len, len(mzs), n_anns, mzrange, tol)
        self.features.MatchScore *= np.sqrt(pep_len)

    def _calculate_sequence_coverage(self, seq_ions) -> int:
        """
        Calculates features related to the sequence coverage by the ion
        annotations.

        Args:
        """
        # The maximum number of fragments annotated by theoretical ions
        # across the charge states. mod is the maximum number of fragments
        # containing the modified residue
        n_anns: int = 0

        # The longest consecutive ion sequence found among charge states
        max_ion_seq_len = 0
        # Across the charge states, the maximum number of each ion type
        # annotated
        max_ion_counts = {"b": -1, "y": -1}

        for _charge in range(2):
            # The number of annotated ions
            n_ions = 0

            for ion_type in ["y", "b"]:
                # A list of b-/y-ion numbers (e.g. 2 for b2[+])
                ion_nums = sorted(
                    [int(ion.split('[')[0][1:]) for ion, t, c, _ in seq_ions
                     if t == ion_type and c == _charge + 1])

                # Increment the number of ions annotated for the current charge
                n_ions += len(ion_nums)

                # Update the max number of annotated ions if appropriate
                if len(ion_nums) > max_ion_counts[ion_type]:
                    max_ion_counts[ion_type] = len(ion_nums)

                ion_seq_len, _ = utilities.longest_sequence(ion_nums)
                if ion_seq_len > max_ion_seq_len:
                    max_ion_seq_len = ion_seq_len

            if n_ions > n_anns:
                n_anns = n_ions

        self.features.NumSeriesbm = max_ion_counts["b"]
        self.features.NumSeriesym = max_ion_counts["y"]

        # The longest sequence tag found divided by the peptide length
        self.features.SeqTagm = max_ion_seq_len

        return n_anns
