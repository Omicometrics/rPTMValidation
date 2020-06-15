import copy
import os
import tempfile
from typing import Optional
import unittest

import numpy as np
from pepfrag import ModSite, Peptide

from rPTMDetermine import PSM, PSMContainer
from rPTMDetermine.readers import PTMDB
from rPTMDetermine.results import write_psm_results


DEFAULT_PEPTIDE = Peptide('AAA', 2, [])


def make_psm(
        data_id: str,
        spec_id: str,
        ml_scores: np.ndarray,
        peptide: Optional[Peptide] = None
) -> PSM:
    if peptide is None:
        peptide = copy.deepcopy(DEFAULT_PEPTIDE)

    psm = PSM(data_id, spec_id, peptide)
    psm.ml_scores = ml_scores
    return psm


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

        write_psm_results(self.container, self.temp_file.name)

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


class TestPSMContainer(unittest.TestCase):
    def test_get_best_psms(self):
        """
        Tests that the PSMContainer.get_best_psms method behaves as expected.

        """
        container = PSMContainer([
            # Data1, Spec1 - both pass consensus, second has higher score
            make_psm('Data1', 'Spec1', np.array([0.1, 0.2, 0.3]),
                     peptide=Peptide('ACK', 2, [])),
            make_psm('Data1', 'Spec1', np.array([0.2, 0.3, 0.4])),  # keep
            # Data1, Spec2 - second PSM passes consensus, first has higher score
            make_psm('Data1', 'Spec2', np.array([-0.1, 0.1, 0.5]),
                     peptide=Peptide('ACK', 2, [])),
            make_psm('Data1', 'Spec2', np.array([0.1, 0.1, 0.1])),  # keep
            # Data1, Spec3 - both fail consensus, second has higher score
            make_psm('Data1', 'Spec3', np.array([-0.3, -0.2, -0.1]),
                     peptide=Peptide('ACK', 2, [])),
            make_psm('Data1', 'Spec3', np.array([-0.2, -0.1, -0.1])),  # keep
            # Data1, Spec4 - only one PSM
            make_psm('Data1', 'Spec4', np.array([-0.3, 0.4, 0.5])),  # keep
        ])

        best_container = container.get_best_psms()

        expected_container = PSMContainer([
            make_psm('Data1', 'Spec1', np.array([0.2, 0.3, 0.4])),
            make_psm('Data1', 'Spec2', np.array([0.1, 0.1, 0.1])),
            make_psm('Data1', 'Spec3', np.array([-0.2, -0.1, -0.1])),
            make_psm('Data1', 'Spec4', np.array([-0.3, 0.4, 0.5])),
        ])

        self.assertEqual(expected_container, best_container)


if __name__ == '__main__':
    unittest.main()
