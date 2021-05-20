import os
import sys
import tqdm
import collections
import numpy as np
from operator import itemgetter
from typing import Optional, Dict, List, Tuple


from rPTMDetermine import PSMContainer
from rPTMDetermine import proteolysis
from rPTMDetermine import similarity
from rPTMDetermine.mass_spectrum import Spectrum
from pepfrag import IonType, Peptide
from rPTMDetermine.peptide_spectrum_match import PSM
from rPTMDetermine.readers import PTMDB


proteolyzer = proteolysis.Proteolyzer('Trypsin')
ptmdb = PTMDB()

DefaultIontype = {
    IonType.b: ["H2O", "NH3"], IonType.y: ["NH3", "H2O"], IonType.a: []
}


class Library():
    """ Construct mass spectrum library for search. """
    def __init__(self, iontype=None):
        if iontype is None:
            iontype = DefaultIontype
        self.DefaultIonType = iontype

    def search(self, mass_spectra: Optional[list, PSMContainer],
               tol: float = 0.1):
        """ Search the library using the mass spectra set. """
        pass

    def _decoy_mass_spectrum(self, ions: Dict[str, (int, int)],
                             spectrum: Spectrum, peptide: Peptide) -> Spectrum:
        """
        Generate decoy mass spectrum by shifting m/z to corresponding
        theoretical ions of reversed peptide sequence.
        """
        # reverse peptide sequence but keeping C-terminal residue if this is
        # tryptic peptides the m/z in mass spectrum is then shifted to the ion
        # m/z of reversed peptides the simplication of this procedure can be
        # simply update the ions.
        dec_pep: Peptide = self._decoy_peptide(peptide)

        # decoy peptide fragment ions
        dec_ions: Dict[str, float] = self._ion_mass(dec_pep)
        # normal peptide fragment ions
        norm_ions: Dict[str, float] = self._ion_mass(peptide)

        # update ion m/z
        spec_peaks = spectrum._peaks.copy()
        for ion, (j, k) in ions.items():
            # if the ion is y1 in a tryptic peptide or precursor ion, keep it
            if (k==1 and "y" in ion) or "M" in ion or ion.startswith("imm"):
                continue

            # charge of the ion
            mz = spec_peaks[j][0]
            mz_err = norm_ions[ion] - mz
            if ion in dec_ions:
                dec_mz = dec_ions[ion] - mz_err
                spec_peaks[j][0] = dec_mz

        return Spectrum(spec_peaks, spectrum.prec_mz, None)

    def _decoy_peptide(self, peptide: Peptide) -> Peptide:
        """ Generate decoy library peptides. """
        if not peptide.mods:
            return Peptide("".join([peptide.seq[:-1][::-1], peptide.seq[-1]]),
                           peptide.charge, [])

        # reverse AA to generate decoys
        list_aa = list(peptide.seq)
        # record modifications and replace corresponding residue by the index
        # of the modification
        term_mods, site_mods = [], collections.defaultdict()
        for i, mod in enumerate(peptide.mods):
            if isinstance(mod.site, str) or mod.site == len(peptide.seq):
                term_mods.append(mod)
            else:
                k = mod.site - 1
                site_mods[f"{i}"] = (list_aa[k], mod)
                list_aa[k] = f"{i}"

        # reverse the sequence except the C-terminus residue
        list_aa[:-1] = list_aa[:-1][::-1]
        mods = []
        for i, a in enumerate(list_aa):
            if a in site_mods:
                a, mod = site_mods[a]
                mods.append(mod._replace(site=i + 1))
                list_aa[i] = a

        return Peptide("".join(list_aa), peptide.charge, mods + term_mods)

    def _ion_mass(self, peptide: Peptide) -> Dict[str, float]:
        """ m/z of theoretical ions for the input peptide. """
        ions = peptide.fragment(ion_types=self.DefaultIonType)
        return dict([(ion, mz) for mz, ion, _ in ions])

    def _group_mz_to_bins(self, mz_list: np.ndarray,
                          tol: float) -> Dict[int, List[int]]:
        """ Group m/z into list for fast search of precursor m/z. """
        mz_round = np.rint(mz_list / tol).astype(int)
        mz_bin_ix: Dict[int, List[int]] = collections.defaultdict(list)
        for i, mz in enumerate(mz_round):
            mz_bin_ix[mz].append(i)

        return mz_bin_ix

    def _bin_frag_mz(self, spectrum: Spectrum,
                     frag_tol: float, check_tol: bool = False)\
            -> Tuple[Dict[int, int], Dict[int, int]]:
        """ Bin fragment ion m/z based on tolerance. """
        # bin m/z
        tmp_mz_bin = np.rint(spectrum.mz / frag_tol).astype(int)
        spec_intens = spectrum.intensity
        # to dictionary
        mz_bin_ix: Dict[int, int] = collections.defaultdict()
        mz_bin_ix2: Dict[int, int] = collections.defaultdict()
        # get index of binned m/z
        for i, (bmz, it) in enumerate(zip(tmp_mz_bin, spec_intens)):
            if bmz not in mz_bin_ix or mz_bin_ix[bmz] < it:
                mz_bin_ix[bmz] = i

        # whether expand the dict by considering the tolerance
        if check_tol:
            for i, (bmz0, it) in enumerate(zip(tmp_mz_bin, spec_intens)):
                for bmz in [bmz0-1, bmz0+1]:
                    if bmz not in mz_bin_ix or mz_bin_ix[bmz] < it:
                        mz_bin_ix2[bmz] = i

        return mz_bin_ix, mz_bin_ix2

    def _get_match_mz_index(self, mzset: Dict[int, int],
                            ref_mzset: Dict[int, int])\
            -> Tuple[List[int], List[int]]:
        """ Match m/z to get indices. """
        common_mz = sorted(set(mzset.keys()) & set(ref_mzset.keys()))
        # return the index
        return ([mzset[mz] for mz in common_mz],
                [ref_mzset[mz] for mz in common_mz])

    def _unique_match(self, index: list, intensity: np.ndarray,
                      ref_index: list, ref_intensity: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray]:
        """ Unique matches between peaks in both mass spectra. """

        def _get_unique_index(index1, index2, intensity2):
            """ Get unique index. """
            if len(set(index1)) == len(index1):
                return index1, index2
            # unique index
            idx = collections.defaultdict(list)
            for j, i in enumerate(index1):
                idx[i].append(index2[j])
            # retain the one with highest intensities
            ix1, ix2 = list(idx.keys()), []
            for i in ix1:
                ixk = idx[i]
                ix2.append(ixk[np.argmax(intensity2[ixk])]
                           if len(ixk) > 1 else ixk[0])
            return ix1, ix2

        idx1, idx2 = _get_unique_index(index, ref_index, ref_intensity)
        idx2, idx1 = _get_unique_index(idx2, idx1, intensity)

        return intensity[idx1], ref_intensity[idx2]

    def _calculate_spectral_similarities_mz(self, target_spectra, ref_spectra_set, prec_tol=0.1, frag_tol=0.2):
        """ Reimplementation of calculate spectra similarities """
        # TODO: similarity score calculation here should be combined with the similarity
        #       score calculation for comparing modified PSMs with their analogues.
        # organize the spectral set
        precursor_mzs = np.array([pmz for _, pmz, _, _ in ref_spectra_set])
        # bin precursors
        blk_ix = self._group_mz_to_bins(precursor_mzs, prec_tol)

        # sort the target spectra
        target_spectra.sort(key=itemgetter(1))

        # calculate similarities
        sim_scores, ref_spectra_bins, min_ref_prec_mz = collections.defaultdict(), collections.defaultdict(), None
        for psm_id, pmz, (ions, spec), _ in tqdm.tqdm(target_spectra, desc="Calculate similarity scores"):
            # round to the nearest integer
            bpmz = int(pmz / prec_tol + 0.5)
            frag_mz_set, mz_tol_set = self._bin_frag_mz(spec, frag_tol, check_tol=True)
            target_mz = set(frag_mz_set.keys())

            # preliminary filering using the common annotations
            match_refs = []
            for bpmz2 in [bpmz - 1, bpmz, bpmz + 1]:
                if bpmz2 not in ref_spectra_bins and bpmz2 in blk_ix:
                    tix = blk_ix[bpmz2]
                    tmp_spec_bins = []
                    for i in tix:
                        id_ref, _, (_, spec_ref), g = ref_spectra_set[i]
                        ref_bin_ix, _ = self._bin_frag_mz(spec_ref, frag_tol)
                        tmp_spec_bins.append((id_ref, ref_bin_ix, set(ref_bin_ix.keys()), spec_ref, g))
                    ref_spectra_bins[bpmz2] = tmp_spec_bins

                if bpmz2 in ref_spectra_bins:
                    match_refs += [(len(ref_frag_bins & target_mz), id_ref, ref_frag_bin_ix, ref_spec, g)
                                   for id_ref, ref_frag_bin_ix, ref_frag_bins, ref_spec, g in ref_spectra_bins[bpmz2]
                                   if id_ref != psm_id]

            if not match_refs:
                sim_scores[psm_id] = (0, None, None)
                continue

            # sort the candidates from highest number of coverage to lowest number
            match_refs.sort(key=itemgetter(0), reverse=True)

            # get candidates and calculate similarity scores
            # :: sqrt root of spectral intensities
            target_int = spec.intensity
            mz_tol_set.update(frag_mz_set)
            # :: similarities
            s = (0, None, None)
            for _, ref_id, ref_frag_bin_ix, ref_spec, g in match_refs[:50]:
                # Get the peak indices of the peaks which match between the two spectra
                idx, ref_idx = self._get_match_mz_index(mz_tol_set, ref_frag_bin_ix)
                # unique peak indice for matching
                if len(set(idx)) < len(idx) or len(set(ref_idx)) < len(ref_idx):
                    intens1, intens2 = self._unique_match(idx, target_int, ref_idx, ref_spec.intensity)
                else:
                    intens1, intens2 = target_int[idx], ref_spec.intensity[ref_idx]
                # Calculate the dot product of the spectral matches
                sk = (intens1 * intens2).sum()
                if sk > s[0]:
                    s = (sk, ref_id, g)

            if psm_id not in sim_scores or sim_scores[psm_id][0] < s[0]:
                sim_scores[psm_id] = s

            # update the mz
            if min_ref_prec_mz is None:
                min_ref_prec_mz = pmz

            if pmz - min_ref_prec_mz > 5:
                del_mzs = [mz for mz in ref_spectra_bins.keys() if pmz - (mz * prec_tol) > 5]
                for dmz in del_mzs:
                    del ref_spectra_bins[dmz]
                min_ref_prec_mz = pmz

        return sim_scores

    def _remove_precursors(self, ions, spectrum, charge):
        """ Remove precursor and neutral loss of precursor peaks, with those around 1Da for isotopic peaks. """
        mz = spectrum.mz
        # precursor m/z with neutral losses should be removed
        pmass = (spectrum.prec_mz - mh) * charge
        retain_peak_idx = np.ones(mz.size, dtype=bool)
        for c in range(1, charge + 1):
            pmz_series = [pmass / charge + mh, (pmass - mh2o) / charge + mh, (pmass - mnh3) / charge + mh]
            for pmz in pmz_series:
                retain_peak_idx[np.absolute(mz - pmz) <= 1] = False

        # remove precursor related peaks
        if not retain_peak_idx.all():
            full_idx = set(range(mz.size))
            # if the peaks are annotated by other types of theoretical ions, keep it
            non_prec_idx = set(j for ion, (j, k) in ions.items() if "M" not in ion)
            del_ix = set(np.where(~retain_peak_idx)[0].tolist())
            del_ix.difference_update(non_prec_idx)
            # retained peak index
            retain_idx = sorted(full_idx - del_ix)
            peaks = spectrum._peaks[retain_idx, :]
            # remove precursors in the ion names and update the index
            ix2 = {j: i for i, j in enumerate(retain_idx)}
            ions = {ion: (ix2[j], k) for ion, (j, k) in ions.items() if "M" not in ion}

            if peaks.size == 0:
                return None, None

            return ions, Spectrum(peaks, spectrum.prec_mz, None)

        return ions, spectrum

    # :: Normalize peaks
    def _normalize_peaks(self, spectrum):
        """ Normalize peaks to squared sum of 1. """
        peaks = spectrum._peaks
        peaks[:, 1] = np.sqrt(peaks[:, 1]) / np.sqrt(peaks[:, 1].sum())
        return Spectrum(peaks, spectrum.prec_mz, None)

    # :: load mass spectra and generate decoy mass spectra
    def _denoise_mass_spectrum(self, psm):
        """ Load mass spectrum and denoise. """
        if psm.spectrum is None:
            psm.spectrum = self._get_spectrum(psm.data_id, psm.spec_id)
        ions, despec = psm.denoise_spectrum()
        # remove precursor peaks
        ions, despec = self._remove_precursors(ions, despec, psm.charge)
        if despec is None:
            return None, None
        # normalize peaks
        despec = self._normalize_peaks(despec)
        return ions, despec
