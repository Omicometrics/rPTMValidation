import unittest

import numpy as np

from rPTMDetermine.mass_spectrum import Spectrum


class TestMassSpectrum(unittest.TestCase):
    def test_spectrum_construction_from_array(self):
        peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 609., 2)

        np.testing.assert_equal(spectrum._peaks, peaks)
        self.assertTrue(spectrum)

    def test_spectrum_construction_from_array_transpose(self):
        peaks = np.array([
                [100., 123., 407., 413.],
                [12., 23., 103., 104.]
            ])
        spectrum = Spectrum(peaks, 609., 2)

        expected = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])

        np.testing.assert_equal(spectrum._peaks, expected)

    def test_spectrum_construction_from_list(self):
        peaks = [
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ]
        spectrum = Spectrum(peaks, 609., 2)

        np.testing.assert_equal(spectrum._peaks, np.array(peaks))

    def test_spectrum_indexing(self):
        peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 609., 2)

        np.testing.assert_equal(spectrum.mz, peaks[:, 0])
        np.testing.assert_equal(spectrum.intensity, peaks[:, 1])

        self.assertEqual(4, len(spectrum))

        self.assertEqual(104., spectrum.raw_base_peak_intensity)
        self.assertEqual(104., spectrum.max_intensity())

        for ii, peak in enumerate(spectrum):
            np.testing.assert_equal(peak, peaks[ii])

        self.assertEqual(100., spectrum[0, 0])

        spectrum[0, 1] = 14.
        self.assertEqual(14., spectrum[0, 1])

        np.testing.assert_equal(
            spectrum.select([1, 3]),
            np.array([[123., 23.], [413., 104.]])
        )
        np.testing.assert_equal(
            spectrum.select([1, 3], [1]),
            np.array([23., 104.])
        )

    def test_spectrum_sorting(self):
        peaks = np.array([
            [123., 23.],
            [407., 103.],
            [100., 12.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 609., 2)

        expected_peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])

        np.testing.assert_equal(spectrum._peaks, expected_peaks)

    def test_spectrum_normalize(self):
        peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 609., 2)

        spectrum.normalize()

        expected_peaks = np.array([
            [100., 0.1153846],
            [123., 0.2211538],
            [407., 0.9903846],
            [413., 1.],
        ])

        np.testing.assert_almost_equal(spectrum._peaks, expected_peaks)

    def test_two_spectra_equal(self):
        peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum1 = Spectrum(peaks, 609., 2)
        spectrum2 = Spectrum(peaks, 609., 2)

        self.assertEqual(spectrum1, spectrum2)

    def test_two_spectra_unequal_same_precursor(self):
        peaks1 = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        peaks2 = np.array([
            [101., 122.],
            [129., 230.],
            [434., 1044.],
            [490., 1954.],
        ])
        spectrum1 = Spectrum(peaks1, 609., 2)
        spectrum2 = Spectrum(peaks2, 609., 2)

        self.assertNotEqual(spectrum1, spectrum2)

    def test_two_spectra_unequal_different_precursor(self):
        peaks = np.array([
            [100., 12.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum1 = Spectrum(peaks, 611., 3)
        spectrum2 = Spectrum(peaks, 609., 2)

        self.assertNotEqual(spectrum1, spectrum2)

    def test_spectrum_not_equal_non_instance(self):
        peaks = np.array([
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 611., 3)

        self.assertNotEqual(spectrum, peaks.tolist())

    def test_remove_itraq(self):
        peaks = np.array([
            [113.11, 120.],
            [115.2, 190.],
            [116.19, 19.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])
        spectrum = Spectrum(peaks, 611., 3)

        spectrum.remove_itraq()

        expected = np.array([
            [115.2, 190.],
            [123., 23.],
            [407., 103.],
            [413., 104.],
        ])

        np.testing.assert_equal(spectrum._peaks, expected)


if __name__ == '__main__':
    unittest.main()
