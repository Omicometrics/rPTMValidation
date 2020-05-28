import unittest

from pepfrag import ModSite

from rPTMDetermine.peptides import merge_seq_mods


class TestMergePeptideSequenceAndModifications(unittest.TestCase):
    def test_merge_list(self):
        sequence = 'ABCYPLMK'
        modifications = [
            ModSite(None, 'N-term', 'iTRAQ8plex'),
            ModSite(None, 1, 'testmod'),
            ModSite(None, 4, 'Nitro'),
            ModSite(None, 'C-term', 'ctermmod')
        ]

        expected = '[iTRAQ8plex]A[testmod]BCY[Nitro]PLMK[ctermmod]'

        self.assertEqual(expected, merge_seq_mods(sequence, modifications))

    def test_merge_string(self):
        sequence = 'ABCYPLMK'
        modifications = 'iTRAQ8plex@N-term;testmod@1;Nitro@4;ctermmod@C-term'

        expected = '[iTRAQ8plex]A[testmod]BCY[Nitro]PLMK[ctermmod]'

        self.assertEqual(expected, merge_seq_mods(sequence, modifications))

    def test_merge_no_mods(self):
        sequence = 'ABCYPLMK'
        self.assertEqual(sequence, merge_seq_mods(sequence, []))


if __name__ == '__main__':
    unittest.main()
