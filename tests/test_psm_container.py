import copy
import os
import tempfile
import unittest

import numpy as np
from pepfrag import ModSite, Peptide

from rPTMDetermine import PSM, PSMContainer
from rPTMDetermine.readers import PTMDB
from rPTMDetermine.results import write_psm_results


class TestPSMContainerFromCSV(unittest.TestCase):
    def setUp(self):
        psm = PSM(
            'TestID', '1.1.1.1',
            Peptide('AAAAA', 2, [ModSite(0.997035, 1, 'Label:15N(1)'),
                                 ModSite(0.997035, 2, 'Label:15N(1)')])
        )
        psm.ml_scores = np.array([0.1, 0.2, 0.3])

        psm2 = copy.deepcopy(psm)
        psm2.spec_id = '1.1.1.2'
        psm2.ml_scores = np.array([-0.1, 0.1, 0.1])
        psm2.site_diff_score = 0.01

        self.container = PSMContainer([
            psm,  # Validated and localized
            psm2,  # Not validated and not localized
        ])

        self.temp_file = tempfile.NamedTemporaryFile(delete=False)

        write_psm_results(self.container, self.temp_file.name, 1.)

    def tearDown(self):
        self.temp_file.close()
        os.remove(self.temp_file.name)

    def test_from_csv(self):
        """
        Tests that PSMContainer reconstructed from file matches the original
        (without unwritten fields such as spectrum).

        """
        container = PSMContainer.from_csv(self.temp_file.name, PTMDB())
        self.assertEqual(self.container, container)


if __name__ == '__main__':
    unittest.main()
