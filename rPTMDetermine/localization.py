"""
This module performs localization of modifications.

"""
import collections
import itertools

import tqdm
import numpy as np

from operator import attrgetter
from typing import Dict, List, Set, Tuple, Sequence

from .peptide_spectrum_match import PSM
from .machinelearning.validation_model import ValidationModel


Fields = ["target", "model_residue", "top_sites", "next_sites",
          "top_score", "next_score", "diff", "is_loc"]
LocInfo = collections.namedtuple(
    "LocInfo", Fields, defaults=(None,) * len(Fields)
)


class PeptideSpectrumMatchLoc(PSM):
    __slots__ = ["localizations"]

    def __init__(self, data_id, spec_id, peptide, localizations=None):
        super().__init__(data_id, spec_id, peptide)
        self.localizations = localizations


class BaseLocalizer:
    """
    Localization of modifications.
    """
    def __init__(self):
        pass

    def get_loc(self,
                isoforms: List[PSM],
                validator: ValidationModel,
                target_residue: str,
                thresholds: Dict[int, float]) -> List[PSM]:
        """
        Localizes modifications.
        Args:
            isoforms: isoforms for localization.
            validator: validation model
            target_residue: target residue target for constructing model
            thresholds: site difference-based score difference thresholds

        Returns:
            list of psms with localization info
        """
        isoforms = validator.validate(isoforms)
        return self._localize(isoforms, target_residue, thresholds)

    def _localize(self, isoforms: List[PSM],
                  target_residue: str,
                  thresholds: Dict[int, float]) -> List[PSM]:
        """ Localize PSMs. """
        loc_mod_psms: List[PSM] = []
        # group the isoforms based on psm uid
        uid_psms = self._group_psms(isoforms)
        for uid in tqdm.tqdm(uid_psms.keys(),
                             desc=f"Localize {target_residue}"):

            # get localizations
            top_psm, loc_mods = self._parse_localizations(uid_psms[uid],
                                                          target_residue,
                                                          thresholds)

            loc = PeptideSpectrumMatchLoc(top_psm.data_id,
                                          top_psm.spec_id,
                                          top_psm.peptide)
            for attr in ["ml_scores", "validation_score", "features"]:
                setattr(loc, attr, getattr(top_psm, attr))
            loc.localizations = loc_mods
            loc_mod_psms.append(loc)

        return loc_mod_psms

    def _parse_localizations(self, psms: Sequence[PSM],
                             target_residue: str,
                             thresholds: Dict[int, float])\
            -> Tuple[PSM, Dict[str, LocInfo]]:
        """ Groups localizations """
        # sites for modifications in isoforms
        target_mod_sites = self._group_mods(psms, target_residue)
        top_psm = max(psms, key=attrgetter("validation_score"))

        # localization
        if psms == 1:
            loc_mods = {mod: LocInfo(model_residue=target_residue,
                                     target=mod,
                                     top_sites=tuple(sites[0]),
                                     top_score=top_psm.validation_score)
                        for mod, sites in target_mod_sites.items() if sites[0]}
            return top_psm, loc_mods

        # for multiple sites
        loc_mods: Dict[str, LocInfo] = collections.defaultdict()
        # sort PSMs based on validation scores
        psms.sort(key=attrgetter("validation_score"), reverse=True)
        top_psm = psms[0]

        # localization of modifications
        for mod, target_sites in target_mod_sites.items():
            top_sites, top_score, alt_sites = self._get_mod_loc_sites(
                mod, target_sites, psms
            )
            # this modification does not target on target residue
            if top_sites is None:
                continue

            mods_info = {"model_residue": target_residue, "target": mod,
                         "top_sites": tuple(top_sites), "top_score": top_score}
            # if alternative site doesn't exist
            if not alt_sites:
                loc_mods[mod] = LocInfo(**mods_info)
                continue

            # next-score sites and score
            next_sites, next_score = alt_sites[0]
            diff_score = top_score - next_score
            diff_site = self._get_site_difference(top_sites, next_sites)
            k = min(2, diff_site - 1)

            # localization info
            loc_mods[mod] = LocInfo(**mods_info,
                                    next_sites=tuple(alt_sites[1:]),
                                    diff=diff_score,
                                    next_score=next_score,
                                    is_loc=diff_score >= thresholds[k])
        return top_psm, loc_mods

    @staticmethod
    def _get_site_difference(top_sites, next_sites):
        """ Site difference between top- and next-score sites. """
        # No. of sites in two isoforms are different, e.g., one is deamidated,
        # while the other is not
        if len(top_sites) != len(next_sites):
            return 30

        if len(top_sites) == 1:
            return abs(top_sites[0] - next_sites[0])
        return sum(abs(i - j) for i, j in zip(top_sites, next_sites))

    @staticmethod
    def _get_mod_loc_sites(mod, sites, psms):
        """ Gets modification localization sites and scores. """
        if all(len(sj) == 0 for sj in sites):
            return None, None, None

        # Deamidation may not be attached to the peptide, but should be
        # considered in localization of deamidation.
        if mod == "Deamidated":
            # get deamidated seq and other modifications
            dm_seq_mods = set()
            for p in psms:
                if any(m.mod == mod for m in p.mods):
                    mods = frozenset(m.mod for m in p.mods if m.mod != mod)
                    dm_seq_mods.add((p.seq, mods))

            # get localization sites for deamidation
            mod_sites: List[Tuple[List, float]] = []
            for sj, p in zip(sites, psms):
                if not sj:
                    mods = frozenset(m.mod for m in p.mods)
                    if (p.seq, mods) in dm_seq_mods:
                        # non-deamidated
                        mod_sites.append(([], p.validation_score))
                else:
                    mod_sites.append((sj, p.validation_score))
        else:
            mod_sites = [(s, p.validation_score) for s, p in zip(sites, psms)
                         if s]

        # top-score isoform for current modification
        top_sites, top_score = mod_sites[0]

        # isoform with alternative sites
        alternative_sites = [(top_sites, None)]
        for sj, score in mod_sites:
            if not any(sj == s for s, _ in alternative_sites):
                alternative_sites.append((sj, score))

        return top_sites, top_score, alternative_sites[1:]

    @staticmethod
    def _group_psms(psms: List[PSM]):
        """ Group psms based on uid. """
        uid_psms: Dict[str, List[PSM]] = collections.defaultdict(list)
        for p in psms:
            uid_psms[p.uid].append(p)
        return uid_psms

    @staticmethod
    def _group_mods(psms: List[PSM], target_residue: str):
        """ Group modifications based on modification name. """
        # sites containing target residue
        seq = psms[0].seq

        # all modifications
        mods = set(itertools.chain(*[[m.mod for m in p.mods] for p in psms]))

        # modification on target residues
        target_mod_sites: Dict[str, List[list]] = collections.defaultdict(list)
        mod_residues: Dict[str, Set[str]] = collections.defaultdict(set)
        for p in psms:
            curr_mod_sites = collections.defaultdict(set)
            for m in p.mods:
                site = (m.site if isinstance(m.site, int)
                        else 0 if m.site == "nterm" else len(seq))
                curr_mod_sites[m.mod].add(site)
                mod_residues[m.mod].add(p.seq[max(0, site-1)])

            # checks whether the modification exists
            for mod in mods:
                target_mod_sites[mod].append(sorted(curr_mod_sites[mod]))

        return {mod: sites for mod, sites in target_mod_sites.items()
                if target_residue in mod_residues[mod]}


class Localizer(BaseLocalizer):
    """
    Localization of modifications.
    Args:
        validators: validation models.
        thresholds: thresholds for localization, should be a
                    dictionary for specifying site-difference
                    based score thresholds.

    """
    def __init__(self,
                 validators: Dict[str, ValidationModel],
                 thresholds: Dict[int, float]):
        super().__init__()
        self.valiators = validators
        self.thresholds = thresholds

    def localize(self, isoforms: Sequence[PSM]) -> List[PSM]:
        """ Localizes modifications in isoforms. """
        residue_psm_idx = self._residue_psm_group(isoforms)

        # localization based on residues
        uid_loc_psms: Dict[str, List[Tuple[str, PSM]]] =\
            collections.defaultdict(list)
        for a in residue_psm_idx.keys():
            target_aa_psms = [isoforms[i] for i in residue_psm_idx[a]]
            loc_psms = self.get_loc(
                target_aa_psms, self.valiators[a], a, self.thresholds
            )
            # group PSMs based on uid
            for p in loc_psms:
                uid_loc_psms[p.uid].append((a, p))

        # generate PSMs with localization of modifications by
        # residue specific models
        return self._update_psms_with_localization(uid_loc_psms)

    @staticmethod
    def _residue_psm_group(psms: Sequence[PSM]) -> Dict[str, List[int]]:
        """
        Groups psms based on residues for localization by indexing

        """
        # get indices of PSM in psms, grouped by uid to integrate isoforms
        # together for localization, with AAs that targeting modifications
        uid_idx: Dict[str, List[int]] = collections.defaultdict(list)
        uid_mod_aas: Dict[str, Set[str]] = collections.defaultdict(set)
        for i, p in enumerate(psms):
            mod_aas = set(p.seq[m.site - 1] if isinstance(m.site, int)
                          else p.seq[0] if m.site == "nterm" else p.seq[-1]
                          for m in p.mods)
            uid_mod_aas[p.uid].update(mod_aas)
            uid_idx[p.uid].append(i)

        # group them based on residue targeting the modification
        residue_psm_index: Dict[str, List[int]] = collections.defaultdict(list)
        for uid in uid_mod_aas.keys():
            for a in uid_mod_aas[uid]:
                residue_psm_index[a] += uid_idx[uid]
        return residue_psm_index

    @staticmethod
    def _update_psms_with_localization(grouped_psms) -> List[PSM]:
        """
        Updates PSMs with localization of modifications using residue
        specific validation models.

        """
        loc_psms: List[PSM] = []
        for uid in grouped_psms.keys():
            # validation scores grouped by residues
            ml_scores: Dict[str, np.ndarray] = collections.defaultdict()
            val_scores: Dict[str, float] = collections.defaultdict()
            # localization info grouped by modification name
            loc_info: Dict[str, List[LocInfo]] = collections.defaultdict(list)
            for i, (a, p) in enumerate(grouped_psms[uid]):
                ml_scores[a] = p.ml_scores
                val_scores[a] = p.validation_score
                for mod in p.localizations.keys():
                    loc_info[mod].append(p.localizations[mod])
            p = max([p for _, p in grouped_psms[uid]],
                    key=attrgetter("validation_score"))
            psm = PeptideSpectrumMatchLoc(p.data_id, p.spec_id, p.peptide,
                                          localizations=loc_info)
            psm.ml_scores = ml_scores
            psm.validation_score = val_scores
            loc_psms.append(psm)

        return loc_psms
