"""
This module performs localization of modifications.

"""
import collections
import itertools
from operator import itemgetter

import tqdm
import numpy as np

from operator import attrgetter
from typing import Dict, List, Set, Tuple, Sequence, Optional

from .peptide_spectrum_match import PSM
from .machinelearning.validation_model import ValidationModel
from .base import LocInfo, ModLocates


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

            loc = PSM(top_psm.data_id, top_psm.spec_id, top_psm.peptide)
            for attr in ["ml_scores", "validation_score", "features"]:
                setattr(loc, attr, getattr(top_psm, attr))
            loc._localizations = loc_mods
            loc_mod_psms.append(loc)

        return loc_mod_psms

    def _parse_localizations(self, psms: Sequence[PSM],
                             target_residue: str,
                             thresholds: Dict[int, float])\
            -> Tuple[PSM, Dict[str, LocInfo]]:
        """ Groups localizations """
        # sort PSMs based on validation scores
        psms.sort(key=attrgetter("validation_score"), reverse=True)

        # sites for modifications in isoforms
        target_mod_sites = self._group_mods(psms, target_residue)
        top_psm = psms[0]

        # localization
        if len(psms) == 1:
            loc_mods = {mod: LocInfo(model_residue=target_residue,
                                     target=mod,
                                     top_sites=tuple(sites[0]),
                                     top_score=top_psm.validation_score)
                        for mod, sites in target_mod_sites.items()}
            return top_psm, loc_mods

        # for multiple sites
        loc_mods: Dict[str, LocInfo] = collections.defaultdict()
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
                                    next_sites=tuple(alt_sites),
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
            mod_sites: List[Tuple[list, float]] = []
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
    def _group_psms(psms: Sequence[PSM]):
        """ Group psms based on uid. """
        uid_psms: Dict[str, List[PSM]] = collections.defaultdict(list)
        for p in psms:
            uid_psms[p.uid].append(p)
        return uid_psms

    @staticmethod
    def _group_mods(psms: Sequence[PSM], target_residue: str):
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
        self.validators = validators
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
                target_aa_psms, self.validators[a], a, self.thresholds
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

    def _update_psms_with_localization(self, grouped_psms) -> List[PSM]:
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
                for mod, loc in p._localizations.items():
                    loc_info[mod].append(loc)

            # determines the localizations
            mod_locates = []
            for _, info in loc_info.items():
                mod_locates.append(self._get_locates(info))

            p = max([p for _, p in grouped_psms[uid]],
                    key=attrgetter("validation_score"))
            psm = PSM(p.data_id, p.spec_id, p.peptide)
            psm.features = p.features
            psm._localizations = loc_info
            psm.mod_locates = mod_locates
            psm.ml_scores = ml_scores
            psm.validation_score = val_scores
            loc_psms.append(psm)

        return loc_psms

    def _get_locates(self, localizations: List[LocInfo]):
        """ Gets modification locates. """
        if len(localizations) == 1:
            return self._mod_locates(localizations)
        else:
            loc_index = collections.defaultdict(list)
            for i, loc in enumerate(localizations):
                loc_index[(loc.is_loc, loc.top_sites)].append(i)

            # consistently localized
            if (not any(t for t, _ in loc_index.keys())
                    or len(set(loc_index.keys())) == 1):
                return self._mod_locates(localizations)

            # use maximum number as the localization
            lk, _ = max([(c, len(ix)) for c, ix in loc_index.items()],
                        key=itemgetter(1))
            return self._mod_locates(localizations, index=loc_index[lk])

    def _mod_locates(self, localizations: List[LocInfo],
                     index: Optional[List[int]] = None) -> ModLocates:
        """ Modification locates. """
        if len(localizations) == 1:
            loc = localizations[0]
            info = {
                "modification": loc.target, "loc_residues": loc.model_residue,
                "top_score": loc.top_score, "sites": loc.top_sites,
                "frac_supports": 1
            }
            if loc.next_sites is None:
                return ModLocates(**info, is_localized=True)

            alt_sites, _ = loc.next_sites[0]
            k = self._get_site_difference(loc.top_sites, alt_sites)
            return ModLocates(
                **info, alternative_score=loc.next_score,
                score_difference=loc.diff, alternative_sites=alt_sites,
                site_difference=k, is_localized=loc.is_loc
            )

        # multiple sites
        if index is None:
            index = list(range(len(localizations)))

        mod_locates = collections.defaultdict(list)
        loc_resid = []
        for i in index:
            loc = localizations[i]
            r, top_sites = loc.model_residue, loc.top_sites
            loc_resid.append(r)
            mod_locates["top_score"].append((r, loc.top_score))
            mod_locates["sites"].append((r, top_sites))
            if loc.next_sites is not None:
                mod_locates["alternative_score"].append((r, loc.next_score))
                mod_locates["score_difference"].append((r, loc.diff))
                alt_sites, _ = loc.next_sites[0]
                k = self._get_site_difference(top_sites, alt_sites)
                mod_locates["alternative_sites"].append((r, alt_sites))
                mod_locates["site_difference"].append((r, k))
        mod_locates = {key: dict(info) for key, info in mod_locates.items()}

        # if localized sites are not supported by at least half of models,
        # reject the localization
        loc = localizations[index[0]]
        is_loc = loc.is_loc
        if is_loc is None:
            is_loc = True
        r = len(index) / len(localizations)
        if is_loc:
            if ((len(localizations) == 2 and r != 1)
                    or (len(localizations) > 2 and r < 0.5)):
                is_loc = False

        return ModLocates(modification=loc.target, is_localized=is_loc,
                          frac_supports=r, loc_residues=tuple(loc_resid),
                          **mod_locates)
