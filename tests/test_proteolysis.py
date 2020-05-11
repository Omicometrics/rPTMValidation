import unittest

from rPTMDetermine.proteolysis import Proteolyzer


class TestProteolyzer(unittest.TestCase):
    def test_trypsin_no_missed_cleavages(self):
        """
        Tests that the correct peptide sequences are generated using no
        missed cleavages.

        """
        proteolyzer = Proteolyzer('Trypsin')

        protein = 'AAAKGHYTMPLYKAGWYMTR'
        peptides = proteolyzer.cleave(protein, num_missed=0)

        self.assertEqual(
            sorted((
                'GHYTMPLYK',
                'AGWYMTR'
            )),
            sorted(peptides)
        )

    def test_trypsin_one_missed_cleavages(self):
        """
        Tests that the correct peptide sequences are generated using one
        missed cleavages.

        """
        proteolyzer = Proteolyzer('Trypsin')

        protein = 'AAAKGHYTMPLYKAGWYMTR'
        peptides = proteolyzer.cleave(protein, num_missed=1)

        self.assertEqual(
            sorted((
                'AAAKGHYTMPLYK',
                'GHYTMPLYK',
                'AGWYMTR',
                'GHYTMPLYKAGWYMTR'
            )),
            sorted(peptides)
        )

    def test_trypsin_two_missed_cleavages(self):
        """
        Tests that the correct peptide sequences are generated using two
        missed cleavages.

        """
        proteolyzer = Proteolyzer('Trypsin')

        protein = 'AAAKGHYTMPLYKAGWYMTR'
        peptides = proteolyzer.cleave(protein, num_missed=2)

        self.assertEqual(
            sorted((
                'AAAKGHYTMPLYK',
                'GHYTMPLYK',
                'AGWYMTR',
                'GHYTMPLYKAGWYMTR',
                'AAAKGHYTMPLYKAGWYMTR'
            )),
            sorted(peptides)
        )

    def test_trypsin_above_length_range(self):
        """
        Tests that peptides above the configured maximum length are not
        returned.

        """
        proteolyzer = Proteolyzer('Trypsin')

        protein = 'AAAAAAAAAAKAAAAAAK'
        peptides = proteolyzer.cleave(protein, len_range=(7, 10))

        self.assertEqual(
            ('AAAAAAK',),
            peptides
        )

    def test_tryptically_cleaved(self):
        """
        Tests that the Proteolyzer can correctly flag tryptic/non-tryptic
        peptides.

        """
        proteolyzer = Proteolyzer('Trypsin')

        self.assertEqual(True, proteolyzer.is_cleaved('ACYMTHGK'))
        self.assertEqual(False, proteolyzer.is_cleaved('MGHPLWY'))

    def test_trypsin_missed_cleavage_count(self):
        """
        Tests that the Proteolyzer can correctly count the number of missed
        cleavage sites in a peptide.

        """
        proteolyzer = Proteolyzer('Trypsin')

        self.assertEqual(0, proteolyzer.count_missed_cleavages('AGHYK'))
        self.assertEqual(1, proteolyzer.count_missed_cleavages('RMTFAYK'))
        self.assertEqual(2, proteolyzer.count_missed_cleavages('RPLMKAFWYK'))


if __name__ == '__main__':
    unittest.main()
