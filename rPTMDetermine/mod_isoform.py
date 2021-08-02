"""
This module generates isoforms for modification localization.

"""
import itertools
import collections
from typing import List, Dict, Tuple, Any, Set, Sequence, Optional

import rPTMDetermine.ionscore as ionscore

from pepfrag import Peptide, ModSite
from .peptides import merge_seq_mods
from .peptide_spectrum_match import PSM
from .readers import PTMDB
from .readers.ptmdb import ModificationNotFoundException

ptmdb = PTMDB()


def _get_special_mod_mass(mod: str) -> Tuple[Optional[float], str]:
    """
    Get special modification mass not in UniMod DB.
    Args:
        mod: name of the modification.

    Returns:
        mass and corrected name of the modification.
    """
    def _get_mass(mod_name):
        try:
            return ptmdb.get_mass(mod_name)
        except ModificationNotFoundException:
            return None

    # In ProteinPilot, if two modifications targets at same sites,
    # it uses the form mod1+mod2, thus the mass should be the total
    # mass of sub-modifications
    if "+" in mod:
        mass = 0
        for name in mod.split("+"):
            if name.endswith("-add"):
                name = name.rstrip("-add")
            mk = _get_mass(name)
            if mk is None:
                return None, name
            mass += mk
        return mass, mod

    # Only consider -add in any else situation
    mod = mod.rstrip("-add")
    return _get_mass(mod), mod


CHARGE_LABELS = [
    ['[+]' if cj == 0 else f'[{cj + 1}+]' for cj in range(charge)]
    for charge in range(10)
]


def _get_ion_score(seq, charge, ions, spectrum_mz, tol):
    """
    Generates ion scores for fast picking up high-quality matches.
    """
    if not ions:
        return 0.

    # sequence coverage
    nseq = max([len([v for v in ions if ck in v])
                for ck in CHARGE_LABELS[charge]])

    return ionscore.ionscore(len(seq), spectrum_mz.size, nseq,
                             spectrum_mz[-1] - spectrum_mz[0], tol)


class Isoform:
    """
     This generates isoforms and performs localization.
     Args:
        sites_in_matches: All sites potentially targeted by the
                          modification. This is obtained from database
                          search results, by summarizing all sites
                          assigned to the modification.
        tol: Tolerance for generating features.
        max_num_sites: Maximum number of alternative sites for each
                       modification in the isoform list. This is used
                       to reduce number of isoforms generated, majority
                       of which are low-score, meaningless in the
                       localization and memory consuming.
     """
    def __init__(self, sites_in_matches: Optional[Dict[str, Set[str]]] = None,
                 tol: float = 0.2, max_num_sites: int = 5):
        self.modification_sites = sites_in_matches
        self.tol = tol
        self.max_num_sites = max_num_sites

        # This is for grouping commonly observed series of residues that
        # are potential sites for a modification if one of the residue in
        # each series is modified, for example, deamidation should be observed
        # at residue N and Q, oxidation is commonly observed at residues H, W,
        # F, Y and P, phosphorylation at S and T, etc.
        self.mod_residue_combines: List[str] = ["DE", "ST", "HWFYP",
                                                "KR", "IL", "NQ"]

    def get_isoforms(self, psm: PSM) -> List[PSM]:
        """
        Gets the isoforms for the PSM.
        Args:
            psm: Modified peptide spectrum match.

        Returns:
            List of isoforms with modifications at different sites.
        """
        if not psm.mods:
            return [psm]

        # Identifies whether mod is in UniMod DB and mass spectrum is included.
        self._valid_check_for_loc(psm)

        mod_info = self._psm_mods(psm.mods, psm.seq)
        pep_isoforms = self._generate_isoforms(psm, mod_info)

        # considers the isoform with non-monoisotopic precursor
        pep_isoforms = self._precursor_correct_isoform(pep_isoforms)
        if not pep_isoforms:
            return pep_isoforms

        # remove invalid isoforms
        pep_isoforms = self._is_valid_mod_psm(pep_isoforms)

        # isoform PSMs
        isoforms = self._pep2psm(psm, pep_isoforms)

        return self._generate_features(isoforms)

    @staticmethod
    def _psm_mods(mods: Sequence[ModSite],
                  seq: str) -> Dict[str, Tuple[float, int, set]]:
        """ Summarizes modifications from the input PSM. """
        # get modifications
        mod_residues: Dict[str, set] = collections.defaultdict(set)
        mod_mass: Dict[str, float] = collections.defaultdict()
        mod_num: Dict[str, int] = collections.defaultdict(int)
        for m in mods:
            mass = m.mass
            if m.mass is None:
                try:
                    mass = ptmdb.get_mass(m.mod)
                except ModificationNotFoundException:
                    raise
            mod_mass[m.mod] = mass
            mod_residues[m.mod].add(
                seq[m.site-1] if isinstance(m.site, int) else ""
            )
            mod_num[m.mod] += 1

        return {mod: (mod_mass[mod], mod_num[mod], sites)
                for mod, sites in mod_residues.items()}

    def _generate_isoforms(self, psm: PSM,
                           psm_mods: Dict[str, Tuple[float, int, set]])\
            -> List[Peptide]:
        """ Generates peptide isoforms. """
        # unmodified PSM as initial point for adding modifications
        isoforms: List[Peptide] = [Peptide(psm.seq, psm.charge, [])]

        # if is deamidated, update the initial isoforms
        if any("Deamidated" in mod for mod in psm_mods.keys()):
            isoforms = self._deamidated_isoform(psm.peptide)
            # clears the deamidated, but keeps modification combinations,
            # e.g., Methyl+Deamidated
            if "Deamidated" in psm_mods:
                del psm_mods["Deamidated"]

        for mod, (mass, n, sites) in psm_mods.items():
            # gets sites
            res, has_nterm, has_cterm = self._get_sites(mod, sites)
            # generates isoforms iteratively
            curr_isos = []
            for iso in isoforms:
                tmp_isos = self._add_mod_isoform(
                    iso, mod, mass, res, n,
                    consider_nterm=has_nterm, consider_cterm=has_cterm
                )
                if tmp_isos is not None:
                    curr_isos += tmp_isos
            isoforms = curr_isos

        return isoforms

    @staticmethod
    def _deamidated_isoform(pep: Peptide) -> List[Peptide]:
        """
        Isoforms for deamidation. As deamidation can be artifically
        introduced due to incorrect selection of monoisotopic peak
        from isotopic distribution of the precursor, not only
        localization, but also correction should be performed.

        """
        tmod, mdm = "Deamidated", 0.984016
        seq, mods, c = pep.seq, pep.mods, pep.charge

        # deamidations and sites
        dm_sites: List[Any] = []
        base_mods: List[ModSite] = []  # base modification
        for m in mods:
            if "Deamidated" in m.mod:
                dm_sites.append(m.site)
                # for modification, e.g., Methyl+Deamidated, remove Deamidated
                if "+" in m.mod:
                    mod = "+".join([s for s in m.mod.split("+") if s != tmod])
                    # update mass
                    mass = m.mass - mdm
                    base_mods.append(ModSite(mass=mass, mod=mod, site=m.site))
            else:
                base_mods.append(m)

        if not dm_sites:
            return [pep]

        # potential sites for deamidation: NQR; deamidation of arginine
        # is named as Citrullination.
        ex_sites = set([m.site if isinstance(m.site, int)
                        else 1 if m.site == "nterm" else len(seq)
                        for m in base_mods])
        dm_sites_tot: List[int] = [i + 1 for i, r in enumerate(seq)
                                   if r in "NQR" and i + 1 not in ex_sites]

        # generate isoforms
        isoforms: List[Peptide] = []
        for k in range(1, len(dm_sites) + 1):
            for ix in itertools.combinations(dm_sites_tot, k):
                alt_mods = [ModSite(mass=mdm, site=i, mod=tmod) for i in ix]
                isoforms.append(Peptide(seq, c, base_mods + alt_mods))

        # no deamidation
        iso_pep = Peptide(seq, c, base_mods)
        isoforms.append(iso_pep)

        return isoforms

    @staticmethod
    def _precursor_correct_isoform(isoforms: Sequence[Peptide])\
            -> List[Peptide]:
        """
        Corrects isoform precursor mass, that mass of modification
        combinations equals to 0 or is close to non-monoisotopic mass
        in isotopic distribution, resulting in potential false
        positives.

        """
        # 13C addition mass comparing to 12C
        m_13c_add = 1.0034
        tol = 0.1
        # corrections
        corr_isoforms: List[Peptide] = []
        for iso in isoforms:
            corr_isoforms.append(iso)
            if len(iso.mods) <= 1:
                continue

            # generate isoforms as potential corrections: up to 3 combinations
            tmp_isoforms = []
            for k in range(2, min(len(iso.mods) + 1, 4)):
                for mods in itertools.combinations(iso.mods, k):
                    mm = sum(m.mass for m in mods)
                    if -0.2 <= mm <= 3.1:
                        n = round(mm / m_13c_add)
                        if abs(mm - n * m_13c_add) <= tol:
                            exp_mods = {(m.mod, m.site) for m in mods}
                            p = Peptide(iso.seq, iso.charge,
                                        [m for m in iso.mods
                                         if (m.mod, m.site) not in exp_mods])
                            tmp_isoforms.append(p)

            if tmp_isoforms:
                corr_isoforms += tmp_isoforms

        return corr_isoforms

    def _get_sites(self, mod: str, sites: Set[str]):
        """
        Get sites for target modification.
        Args:
            mod: modification name in UniMod database name.
            sites: Potential sites for the modification assigned by
                   database search.

        Returns:
            Sites for the modification.
        """
        # sites in database
        try:
            db_sites = ptmdb.get_mod_sites(mod)
        except ModificationNotFoundException:
            db_sites = set()

        # determines the N- and C- terminus from database search results, if
        # isn't assigned, use the sites defined in Unimod database only.
        exp_sites = None
        if self.modification_sites is not None:
            exp_sites = self.modification_sites[mod]
            has_nterm = "nterm" in exp_sites
            has_cterm = "cterm" in exp_sites
        else:
            has_nterm = "N-term" in db_sites
            has_cterm = "C-term" in db_sites

        if not sites:
            if exp_sites is not None and has_nterm and "K" in exp_sites:
                return "K", has_nterm, has_cterm
            return "", has_nterm, has_cterm

        # combination of targets as potential sites
        target_comb = ("".join([rx for rx in self.mod_residue_combines
                                if any(site in rx for site in sites)])
                       + "".join(sites))
        valid_sites = db_sites.intersection(target_comb)
        valid_sites.update(sites)  # compensate for ones assigned by database

        return "".join(valid_sites), has_nterm, has_cterm

    @staticmethod
    def _add_mod_isoform(pep: Peptide, mod: str, mod_mass: float,
                         mod_residues: str, nmod: int,
                         consider_nterm: bool = False,
                         consider_cterm: bool = False)\
            -> Optional[List[Peptide]]:
        """ Generate isoforms for localization. """
        # identify whether multiple target residue exists
        base_mods = [m for m in pep.mods if m.mod != mod]
        base_sites = set([m.site for m in base_mods])
        res_sites = [i + 1 for i, r in enumerate(pep.seq)
                     if r in mod_residues and i + 1 not in base_sites]

        # terminus
        if consider_nterm and not ("nterm" in base_sites or 1 in base_sites):
            if 1 not in res_sites:
                res_sites.insert(0, "nterm")

        if consider_cterm and not ("cterm" in base_sites
                                   or len(pep.seq) in base_sites):
            if len(pep.seq) not in res_sites:
                res_sites.append("cterm")

        if len(res_sites) < nmod:
            return None

        # generate isoforms
        isoforms: List[Peptide] = []
        for ix in itertools.combinations(res_sites, nmod):
            loc_mods = [ModSite(mass=mod_mass, site=k, mod=mod) for k in ix]
            iso_pep = Peptide(pep.seq, pep.charge, loc_mods + base_mods)
            isoforms.append(iso_pep)

        return isoforms

    def _pep2psm(self, psm: PSM, isoforms: Sequence[Peptide]) -> List[PSM]:
        """
        Converts Peptide to PSMs
        """
        iso_psms: List[PSM] = []
        iso_peps: Set[str] = set()
        for iso in isoforms:
            pk = merge_seq_mods(iso.seq, iso.mods)
            if pk not in iso_peps:
                iso_psm = PSM(psm.data_id, psm.spec_id, iso,
                              spectrum=psm.spectrum)
                iso_psms.append(iso_psm)
                iso_peps.add(pk)

        # supposes 5 modification types with each of 5 alternative sites
        if len(iso_psms) <= self.max_num_sites * 3:
            return iso_psms

        return self._filter_isoform(iso_psms)

    def _filter_isoform(self, psms: Sequence[PSM]) -> List[PSM]:
        """
        Filters out isoforms with low ion scores so that each
        modification has no more than `self.max_num_sites` alternative sites
        for subsequent localization.

        """
        # scores for filtering low-quality or redundant isoforms
        scores: List[float] = []
        mod_sits: List[Dict[str, Set[Any]]] = []
        for p in psms:
            # Denoising
            ions, despec = p.denoise_spectrum(tol=self.tol)
            # Calculates score
            _ions = [ion for ion in ions.keys() if ion[0] in "yb"]
            score = _get_ion_score(p.seq, p.charge, _ions, despec.mz, self.tol)
            scores.append(score)
            # collects modification types and No. of sites for filtering
            tmp_sites = collections.defaultdict(set)
            for m in p.mods:
                tmp_sites[m.mod].add(m.site)
            mod_sits.append({mod: frozenset(sites)
                             for mod, sites in tmp_sites.items()})

        # modifications and sites
        all_mod_sites: Dict[str, Set[frozenset]] = collections.defaultdict(set)
        for mods in mod_sits:
            for mod, sites in mods.items():
                all_mod_sites[mod].add(sites)
        all_mods = set(all_mod_sites.keys())
        num_mod_sites = {m: len(sites) for m, sites in all_mod_sites.items()}

        # sorts scores
        six = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        # top-score isoforms, with at least 3 alternative sites for each mod.
        retain_index: List[int] = []
        num_sites_cum: Dict[str, int] = {m: 0 for m in all_mods}
        mod_sites_cum: Dict[str, Set[frozenset]] = collections.defaultdict(set)
        for i in six:
            tmp_sites = mod_sits[i]
            # combine modification based on sites and add up the sites if new
            # site is found
            t = False
            for mod in all_mods:
                sites = tmp_sites[mod] if mod in tmp_sites else frozenset([])
                if sites not in mod_sites_cum[mod]:
                    # this is a new site and No. of alternative sites <= 3
                    if num_sites_cum[mod] < self.max_num_sites:
                        t = True
                    num_sites_cum[mod] += 1
                    mod_sites_cum[mod].add(frozenset(sites))

            # if the new site added is valid, i.e., increases a new site
            # to any modification with No. of sites < 3
            if t:
                retain_index.append(i)

            # No. of sites reaches 3 or maximum for that modification
            if all(n >= self.max_num_sites or n == num_mod_sites[mod]
                   for mod, n in num_sites_cum.items()):
                break

        return [psms[i] for i in retain_index]

    @staticmethod
    def _generate_features(psms: Sequence[PSM]) -> List[PSM]:
        """ Get features for the isoforms. """
        for p in psms:
            p.extract_features()
            p.spectrum = None
        return psms

    @staticmethod
    def _is_valid_mod_psm(psms: Sequence[Any]) -> List[Any]:
        """
        justifies whether the assignments of modifications are valid,
        mainly due to the residue in sites whereas in position the
        modification should be at peptide terminus.

        Args:
            psms: Modified psms or peptides.

        Returns:
            PSMs with valid and corrected modification assignments.

        """
        # all modifications
        mods = set(m.mod for p in psms for m in p.mods)

        # the modifications only happen at peptide terminus
        check_mods: Set[str] = set()
        term_mods: Dict[str, Set[str]] = collections.defaultdict(set)
        for mod in mods:
            positions = ptmdb.get_mod_positions(mod)
            if "Anywhere" not in positions:
                check_mods.add(mod)

            # modification sites
            sites = ptmdb.get_mod_terminus(mod)
            if sites is not None:
                term_mods[mod] = sites

        # remove the modifications not at peptide terminus
        i0, i1 = 1, len(psms[0].seq)
        del_ix: Set[int] = set()
        for i, p in enumerate(psms):
            if any(m.mod in check_mods and isinstance(m.site, int)
                   and i0 < m.site < i1 for m in p.mods):
                del_ix.add(i)
            else:
                # correct the modification sites
                pep_mods = []
                for m in p.mods:
                    if (isinstance(m.site, str) or i0 < m.site < i1
                            or m.mod not in term_mods):
                        pep_mods.append(m)
                    else:
                        if "N-term" in term_mods[m.mod] and m.site == i0:
                            pep_mods.append(ModSite(mod=m.mod, mass=m.mass,
                                                    site="nterm"))
                        elif "C-term" in term_mods[m.mod] and m.site == i1:
                            pep_mods.append(ModSite(mod=m.mod, mass=m.mass,
                                                    site="cterm"))
                        else:
                            pep_mods.append(m)
                p.mods = pep_mods

        if not del_ix:
            return psms

        return [p for i, p in enumerate(psms) if i not in del_ix]

    @staticmethod
    def _valid_check_for_loc(psm: PSM):
        """ Justify whether the psm is suitable for localization. """
        # whether the modification exists in Unimod DB
        for i, m in enumerate(psm.mods):
            if m.mass is None:
                mass, mod = _get_special_mod_mass(m.mod)
                if mass is None:
                    raise ModificationNotFoundException
                psm.mods[i] = m._replace(mass=mass, mod=mod)

        # spectrum must be assigned for getting features
        if psm.spectrum is None:
            raise ValueError("Mass spectrum is not assigned.")
