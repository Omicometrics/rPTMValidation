import copy
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from pepfrag import ModSite, Peptide

from rPTMDetermine import PSM, PSMContainer
from rPTMDetermine.results import write_psm_results


class TestPSMResultsWriter(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self) -> None:
        self.temp_file.close()
        os.remove(self.temp_file.name)

    def test_output(self):
        psm = PSM(
            'TestID', '1.1.1.1',
            Peptide('AAAAA', 2, (ModSite(12.01, 1, 'testmod1'),
                                 ModSite(14.07, 2, 'testmod2')))
        )
        psm.ml_scores = np.array([0.1, 0.2, 0.3])

        psm2 = copy.deepcopy(psm)
        psm2.spec_id = '1.1.1.2'
        psm2.ml_scores = np.array([-0.1, 0.1, 0.1])
        psm2.site_diff_score = 0.01

        container = PSMContainer([
            psm,  # Validated and localized
            psm2,  # Not validated and not localized
        ])

        write_psm_results(container, self.temp_file.name, 1.)

        df = pd.read_csv(self.temp_file.name)
        expected = pd.DataFrame(
            {
                'DataID': ['TestID', 'TestID'],
                'SpectrumID': ['1.1.1.1', '1.1.1.2'],
                'Sequence': ['AAAAA', 'AAAAA'],
                'Charge': [2, 2],
                'Modifications': ['testmod1@1;testmod2@2',
                                  'testmod1@1;testmod2@2'],
                'PassesConsensus': [True, False],
                'PassesMajority': [True, True],
                'Localized': [True, False],
                'Scores': ['0.1;0.2;0.3', '-0.1;0.1;0.1'],
                'SiteScore': [np.NaN, np.NaN],
                'SiteProbability': [np.NaN, np.NaN],
                'SiteDiffScore': [np.NaN, 0.01]
            }
        )

        pd.testing.assert_frame_equal(expected, df)


if __name__ == '__main__':
    unittest.main()
