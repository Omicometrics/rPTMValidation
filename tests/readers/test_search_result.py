import dataclasses
import pickle
import tempfile
import unittest

from pepfrag import ModSite

from rPTMDetermine.readers import PeptideType, SearchResult


@dataclasses.dataclass(eq=True, frozen=True)
class TestSearchResult(SearchResult):
    __slots__ = ('newfield',)

    newfield: str


class TestSearchResultPickling(unittest.TestCase):
    def test_search_result_picklable(self):
        """
        Tests that SearchResult can be pickled and unpickled.

        """
        result = SearchResult(
            seq='AAA',
            mods=(ModSite(23.01, 1, 'testmod')),
            charge=2,
            spectrum='1.1.5',
            dataset=None,
            rank=1,
            pep_type=PeptideType.normal,
            theor_mz=7.
        )

        with tempfile.TemporaryFile() as fh:
            pickle.dump(result, fh)
            fh.seek(0)
            new_result = pickle.load(fh)

        self.assertEqual(result, new_result)

    def test_search_result_subclass_picklable(self):
        """
        Tests that a subclass of SearchResult can be pickled and unpickled.

        """
        result = TestSearchResult(
            seq='AAA',
            mods=(ModSite(23.01, 1, 'testmod')),
            charge=2,
            spectrum='1.1.5',
            dataset=None,
            rank=1,
            pep_type=PeptideType.normal,
            theor_mz=7.,
            newfield='test'
        )

        with tempfile.TemporaryFile() as fh:
            pickle.dump(result, fh)
            fh.seek(0)
            new_result = pickle.load(fh)

        self.assertEqual(result, new_result)


if __name__ == '__main__':
    unittest.main()
