import unittest
from typing import List

from pepfrag import ModSite, Peptide

from rPTMDetermine import PSM
from rPTMDetermine.localization import (
    generate_alternative_nterm_candidates,
    generate_deamidation_candidates,
    generate_localization_isoforms
)
from rPTMDetermine.readers import PTMDB


def make_modified_psm(
        seq: str,
        charge: int,
        mods: List[ModSite],
        target_mod: str,
        target_mod_mass: float,
        target_mod_sites: List[int]
):
    """
    Builds a PSM with a peptide with the configured modification sites.

    """
    mods += [ModSite(target_mod_mass, s, target_mod) for s in target_mod_sites]
    mods.sort(key=lambda ms: ms.site if isinstance(ms.site, int) else 0)
    return PSM(None, None, Peptide(seq, charge, mods))


def make_deamidated_psm(
        seq: str,
        charge: int,
        mods: List[ModSite],
        deam_sites: List[int]
) -> PSM:
    """
    Builds a PSM with a peptide with the configured `deam_sites`.

    """
    return make_modified_psm(
        seq, charge, mods, 'Deamidated', 0.984016, deam_sites
    )


class TestGenerateDeamidationCandidates(unittest.TestCase):
    def setUp(self):
        self.ptmdb = PTMDB()

    def test_no_deamidation(self):
        psm = PSM(
            None,
            None,
            Peptide('AAAAAAK', 2, [])
        )
        self.assertEqual(
            [],
            generate_deamidation_candidates(psm, self.ptmdb)
        )

    def test_one_deamidation_no_alternative_sites(self):
        seq = 'AANAAYK'
        psm = make_deamidated_psm(seq, 2, [], [3])
        self.assertEqual(
            [
                PSM(None, None, Peptide('AANAAYK', 2, [])),
                psm
            ],
            generate_deamidation_candidates(psm, self.ptmdb)
        )

    def test_one_deamidation_two_possible_sites(self):
        seq = 'AANAQNAYK'
        psm = make_deamidated_psm(seq, 2, [], [3])
        self.assertEqual(
            [
                PSM(None, None, Peptide(seq, 2, [])),
                psm,
                make_deamidated_psm(seq, 2, [], [5]),
                make_deamidated_psm(seq, 2, [], [6])
            ],
            generate_deamidation_candidates(psm, self.ptmdb)
        )

    def test_two_deamidations_three_possible_sites(self):
        seq = 'AANAQNAYK'
        psm = make_deamidated_psm(seq, 2, [], [3, 5])
        self.assertEqual(
            [
                PSM(None, None, Peptide(seq, 2, [])),
                make_deamidated_psm(seq, 2, [], [3]),
                make_deamidated_psm(seq, 2, [], [5]),
                make_deamidated_psm(seq, 2, [], [6]),
                psm,
                make_deamidated_psm(seq, 2, [], [3, 6]),
                make_deamidated_psm(seq, 2, [], [5, 6]),
            ],
            generate_deamidation_candidates(psm, self.ptmdb)
        )


class TestGenerateLocalizationIsoforms(unittest.TestCase):
    def setUp(self):
        self.ptmdb = PTMDB()

    def test_no_alternative_sites(self):
        nitro_mass = self.ptmdb.get_mass('Nitro')
        psm = make_modified_psm('AAAYK', 2, [], 'Nitro', nitro_mass, [4])
        self.assertEqual(
            [psm],
            generate_localization_isoforms(psm, 'Nitro', nitro_mass, 'Y')
        )

    def test_one_mod_one_alternative_site(self):
        nitro_mass = self.ptmdb.get_mass('Nitro')
        seq = 'AAAYYK'
        psm = make_modified_psm(seq, 2, [], 'Nitro', nitro_mass, [4])
        self.assertEqual(
            [
                psm,
                make_modified_psm(seq, 2, [], 'Nitro', nitro_mass, [5])
            ],
            generate_localization_isoforms(psm, 'Nitro', nitro_mass, 'Y')
        )

    def test_one_mod_two_alternative_sites(self):
        nitro_mass = self.ptmdb.get_mass('Nitro')
        seq = 'AYAAYYK'

        def make_nitro_psm(sites: List[int]) -> PSM:
            return make_modified_psm(seq, 2, [], 'Nitro', nitro_mass, sites)

        psm = make_nitro_psm([5])
        self.assertEqual(
            [
                make_nitro_psm([2]),
                psm,
                make_nitro_psm([6])
            ],
            generate_localization_isoforms(psm, 'Nitro', nitro_mass, 'Y')
        )

    def test_two_mod_one_alternative_site(self):
        nitro_mass = self.ptmdb.get_mass('Nitro')
        seq = 'AYAAYYK'

        def make_nitro_psm(sites: List[int]) -> PSM:
            return make_modified_psm(seq, 2, [], 'Nitro', nitro_mass, sites)

        psm = make_nitro_psm([2, 5])
        self.assertEqual(
            [
                psm,
                make_nitro_psm([2, 6]),
                make_nitro_psm([5, 6]),
            ],
            generate_localization_isoforms(psm, 'Nitro', nitro_mass, 'Y')
        )


class TestGenerateNTermCandidates(unittest.TestCase):
    def test_site_occupied(self):
        """
        Tests that no candidates are generated when the N-terminus is
        already modified.

        """
        psm = make_modified_psm(
            'AYAAK', 2, [ModSite(304.20536, 'nterm', 'iTRAQ8plex')],
            'Nitro', 44.985078, [2]
        )

        self.assertEqual(
            [],
            generate_alternative_nterm_candidates(
                psm, 'Nitro', 'Carbamyl', 43.005814
            )
        )

    def test_site_unoccupied(self):
        """
        Tests that a candidate is generated when the N-terminus is not modified.

        """
        psm = make_modified_psm(
            'AYAAK', 2, [],
            'Nitro', 44.985078, [2]
        )

        expected_psm = PSM(
            None, None,
            Peptide('AYAAK', 2, [ModSite(43.005814, 'nterm', 'Carbamyl')])
        )

        self.assertEqual(
            [expected_psm],
            generate_alternative_nterm_candidates(
                psm, 'Nitro', 'Carbamyl', 43.005814
            )
        )


if __name__ == '__main__':
    unittest.main()
