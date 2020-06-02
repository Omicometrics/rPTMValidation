import unittest

import numpy as np
from pepfrag import ModSite, Peptide

from rPTMDetermine.mass_spectrum import Spectrum
from rPTMDetermine.peptide_spectrum_match import PSM, SpectrumNotFoundError


class TestPSM(unittest.TestCase):
    def test_underlying_peptide(self):
        psm = PSM(None, None, Peptide('AAAK', 2, []))

        self.assertEqual('AAAK', psm.seq)
        self.assertEqual(2, psm.charge)
        self.assertEqual([], psm.mods)

        psm.seq = 'AAALK'
        self.assertEqual('AAALK', psm.seq)
        self.assertEqual('AAALK', psm.peptide.seq)

        new_mods = [ModSite(12.01, 1, 'testmod')]
        psm.mods = new_mods
        self.assertEqual(new_mods, psm.mods)
        self.assertEqual(new_mods, psm.peptide.mods)

    def test_add_spectrum(self):
        psm = PSM(None, None, Peptide('AAAK', 2, []))
        spectrum = Spectrum(np.array([[112., 1.], [145., 3.]]), 413., 2)
        psm.spectrum = spectrum

        self.assertEqual(spectrum, psm.spectrum)

    def test_add_spectrum_non_instance(self):
        psm = PSM(None, None, Peptide('AAAK', 2, []))

        with self.assertRaises(TypeError):
            psm.spectrum = np.array([[112., 1.], [145., 3.]])

    def test_psms_equal(self):
        psm1 = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))
        psm2 = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))

        self.assertEqual(psm1, psm2)

    def test_psms_not_equal_spectrum_id(self):
        psm1 = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))
        psm2 = PSM('testdata', '1.1.1.2', Peptide('AAAK', 2, []))

        self.assertNotEqual(psm1, psm2)

    def test_psms_not_equal_data_id(self):
        psm1 = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))
        psm2 = PSM('testdata1', '1.1.1.1', Peptide('AAAK', 2, []))

        self.assertNotEqual(psm1, psm2)

    def test_psms_not_equal_peptide_sequence(self):
        psm1 = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))
        psm2 = PSM('testdata1', '1.1.1.1', Peptide('AAALK', 2, []))

        self.assertNotEqual(psm1, psm2)

    def test_psms_not_equal_non_instance(self):
        psm = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))

        self.assertNotEqual(psm, ['testdata'])

    def test_spectrum_initialized(self):
        psm = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))

        with self.assertRaises(SpectrumNotFoundError):
            psm._check_spectrum_initialized()

        psm.spectrum = Spectrum(np.array([[112., 1.], [145., 3.]]), 413., 2)

        psm._check_spectrum_initialized()

    def test_is_localized(self):
        psm = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))

        self.assertEqual(True, psm.is_localized())

        psm.site_diff_score = 0.01
        self.assertEqual(False, psm.is_localized())

        psm.site_diff_score = 0.11
        self.assertEqual(True, psm.is_localized())

    def test_extract_features(self):
        psm = PSM('testdata', '1.1.1.1', Peptide('AAAK', 2, []))
        psm.spectrum = Spectrum(np.array([
            # immonium A
            [44.05, 24.],
            # b1-NH3
            [55.02, 13.],
            # b1
            [72.04, 45.],
            # immonium K
            [101.1, 43.],
            # c1
            [129.08, 31.],
            # b2
            [143.08, 67.],
            # y1
            [147.1, 60.],
            # b3
            [214.1, 45.]
        ]), 430., 2)

        psm.extract_features()
        # TODO: more specific assertion
        self.assertTrue(len(psm.features.to_list()))

    def test_extract_features_no_matching_ions(self):
        """
        """


if __name__ == '__main__':
    unittest.main()
