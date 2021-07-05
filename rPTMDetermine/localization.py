"""
This module generates isoforms for localization of modifications.

"""
import itertools
import collections
from operator import itemgetter
from typing import List, Optional, Dict, Tuple, Any

from .features import extract_full_features

from pepfrag import Peptide, ModSite
from rPTMDetermine.peptide_spectrum_match import PSM
from rPTMDetermine.readers import PTMDB
from rPTMDetermine.readers.ptmdb import ModificationNotFoundException

ptmdb = PTMDB()


def _is_valid_mod_psm(psms: List[Any]) -> List[Any]:
    """
    Rules for justifying whether the assignments of modifications
    are valid.
    Args:
        psms: Modified psms.

    Returns:
        Psms with valid modification assignments.
    """
    psm = psms[0]
    del_ix = set()
    # Rule 1: if Gln->pyro-Glu or Glu->pyro-Glu is assigned, it is fixed at
    #         peptide N-terminus, and other modifications shouldn't appear at
    #         there, otherwise is rejected.
    check_mods = {"Gln->pyro-Glu", "Glu->pyro-Glu"}
    if any(m.mod in check_mods for m in psm.mods):
        for i, p in enumerate(psms):
            for m in p.mods:
                if (m.mod in check_mods
                        and (m.site != "nterm"
                             or any(isinstance(m2.site, int)
                                    and m2.site == 1 for m2 in p.mods))):
                    del_ix.add(i)
                    break
    if not del_ix:
        return psms
    return [p for i, p in enumerate(psms) if i not in del_ix]


def _combine_mods(seq: str, mods: List[ModSite]) -> str:
    """ Combine modifications into sequence """
    if not mods:
        return seq

    frags, cterm, mods_ = [], None, []
    for mod in mods:
        # check terminals
        if isinstance(mod.site, str):
            if mod.site == "nterm" or mod.site == "N-term":
                frags.append(f"[{mod.mod}]")
            else:
                cterm = f"[{mod.mod}]"
        else:
            mods_.append(mod)

    mods_ = sorted(mods_, key=itemgetter(1))
    i = 0
    for mod in mods_:
        frags.append(f"{seq[i:mod.site]}[{mod.mod}]")
        i = mod.site

    if i < len(seq): frags.append(seq[i:])
    if cterm is not None: frags.append(cterm)

    return "".join(frags)


class Localization:
    """
     This generates isoforms and performs localization.
     Args:
        sites_in_matches: All sites potentionally targeted by the
                          modification. This is obtained from database
                          search results, by summarizing all sites
                          assigned to the modification.
     """
    def __init__(self, sites_in_matches: Dict[str, set]):
        self.modification_sites = sites_in_matches

        # This is for grouping commonly observed series of residues that
        # are potential sites for a modification if one of the residue in
        # each series is modified, for example, deamidation should be observed
        # at residue N and Q, oxidation is commonly observed at residues H, W,
        # F, Y and P, phosphorylation at S and T.
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
        # check whether it is valid
        pep_isoforms = _is_valid_mod_psm(pep_isoforms)

        # generate isoforms and get features
        return self._get_features(psm, pep_isoforms)

    @staticmethod
    def _psm_mods(mods: List[ModSite],
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

    def _generate_isoforms(self,
                           psm: PSM,
                           psm_mods: Dict[str, Tuple[float, int, set]])\
            -> List[Peptide]:
        """ Generates peptide isoforms. """
        # unmodified PSM as initial point for adding modifications
        isoforms = [Peptide(psm.seq, psm.charge, [])]
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
                if tmp_isos is None:
                    tmp_isos = [iso]
                curr_isos += tmp_isos
            isoforms = curr_isos
        return isoforms

    def _get_sites(self, mod: str, sites: set):
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
        db_sites = ptmdb.get_mod_sites(mod)
        exp_sites = self.modification_sites[mod]

        has_nterm = "nterm" in exp_sites
        has_cterm = "cterm" in exp_sites
        if not sites:
            if has_nterm and "K" in exp_sites:
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
    def _add_mod_isoform(
            pep: Peptide, mod: str, mod_mass: float,
            mod_residues: str, nmod: int,
            consider_nterm: bool = False, consider_cterm: bool = False
    ) -> Optional[List[Peptide]]:
        """ Generate isoforms for localization. """
        # identify whether multiple target residue exists
        base_mods = [m for m in pep.mods if m.mod != mod]
        non_mod_sites = set([m.site for m in base_mods])
        res_sites: List[Any] = [
            i+1 for i, r in enumerate(pep.seq)
            if r in mod_residues and i+1 not in non_mod_sites
        ]

        # consider N-terminal modification
        if consider_nterm and "nterm" not in non_mod_sites:
            res_sites.insert(0, "nterm")

        # consider C-terminal modification
        if consider_cterm and "cterm" not in non_mod_sites:
            res_sites.append("cterm")

        if len(res_sites) < nmod:
            return None

        # generate isoforms
        isoforms = []
        for ix in itertools.combinations(res_sites, nmod):
            loc_mods = [ModSite(mass=mod_mass, site=k, mod=mod) for k in ix]
            iso_pep = Peptide(pep.seq, pep.charge, loc_mods + base_mods)
            isoforms.append(iso_pep)

        return isoforms

    @staticmethod
    def _get_features(psm: PSM, isoforms: List[Peptide]) -> List[PSM]:
        """ Get features for the isoforms. """
        iso_psms, iso_peps = [], set()
        for iso in isoforms:
            pk = _combine_mods(iso.seq, iso.mods)
            if pk not in iso_peps:
                iso_psm = PSM(psm.data_id, psm.spec_id, iso,
                              spectrum=psm.spectrum)
                iso_psm = extract_full_features(iso_psm)
                iso_psms.append(iso_psm)
                iso_peps.add(pk)
        return iso_psms

    @staticmethod
    def _valid_check_for_loc(psm: PSM):
        """ Justify whether the psm is suitable for localization. """
        # whether the modification exists in Unimod DB
        for m in psm.mods:
            if m.mass is None:
                try:
                    _ = ptmdb.get_mass(m.mod)
                except ModificationNotFoundException:
                    raise

        # spectrum must be assigned for getting features
        if psm.spectrum is None:
            raise ValueError("Mass spectrum is not assigned.")
