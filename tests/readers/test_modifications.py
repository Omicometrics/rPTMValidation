import unittest

from pepfrag import ModSite

from rPTMDetermine.readers import PTMDB
from rPTMDetermine.readers.modifications import (
    parse_mods,
    UnknownModificationException
)


class TestModificationParsing(unittest.TestCase):
    def setUp(self):
        self.ptmdb = PTMDB()

    def test_at_format(self):
        """
        Tests that a semi-colon separated modification string, using
        MOD@SITE format, is correctly parsed, including explicit "No"
        modifications.

        """
        mod_str = 'iTRAQ8plex@N-term;Nitro@9;iTRAQ8plex@15;No iTRAQ8plex@18'
        mods = parse_mods(mod_str, self.ptmdb)
        self.assertEqual(
            [
                ModSite(304.20536, 'nterm', 'iTRAQ8plex'),
                ModSite(44.985078, 9, 'Nitro'),
                ModSite(304.20536, 15, 'iTRAQ8plex')
            ],
            mods
        )

    def test_at_format_unknown_modification(self):
        """
        Tests that an exception is raised when an unknown modification is
        detected from modifications in MOD@SITE format.

        """
        mod_str = 'iTRAQ8plex@N-term;TestMod@1'
        with self.assertRaisesRegex(
                UnknownModificationException, r'.*TestMod.*'
        ):
            parse_mods(mod_str, self.ptmdb)

    def test_at_format_delta_modification(self):
        """
        Tests that Delta modifications are correctly parsed using the MOD@SITE
        format.

        """
        mod_str = 'Delta:H(4)C(2)(H)@1'
        mods = parse_mods(mod_str, self.ptmdb)
        self.assertEqual(
            [
                ModSite(28.0313, 1, 'Delta:H(4)C(2)')
            ],
            mods
        )

    def test_bar_format(self):
        """
        Tests that a comma separated modification string, using
        MASS|SITE|MOD format, is correctly parsed.

        """
        mod_str = (
            '304.20536|nterm|iTRAQ8plex,44.985078|9|Nitro,'
            '304.20536|15|iTRAQ8plex'
        )
        mods = parse_mods(mod_str, self.ptmdb)
        self.assertEqual(
            [
                ModSite(304.20536, 'nterm', 'iTRAQ8plex'),
                ModSite(44.985078, 9, 'Nitro'),
                ModSite(304.20536, 15, 'iTRAQ8plex')
            ],
            mods
        )


if __name__ == '__main__':
    unittest.main()
