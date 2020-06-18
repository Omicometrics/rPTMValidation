import os
import tempfile
from types import MappingProxyType
from typing import Any
import unittest

import numpy as np
from pepfrag import MassType, ModSite, Peptide
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from rPTMDetermine.features import Features
from rPTMDetermine.machinelearning.classification import Model
from rPTMDetermine.mass_spectrum import Spectrum
from rPTMDetermine.packing import fullname, save_to_file, load_from_file
from rPTMDetermine.peptide_spectrum_match import PSM
from rPTMDetermine.psm_container import PSMContainer
from rPTMDetermine.readers import (
    PeptideType, ProteinPilotSearchResult, SearchResult
)
from rPTMDetermine.results import write_psm_results
from rPTMDetermine.validation_model import ValidationModel


class TestFullname(unittest.TestCase):
    def test_classes(self):
        classes = {
            'rPTMDetermine.machinelearning.classification.Model': Model,
            'rPTMDetermine.psm_container.PSMContainer': PSMContainer,
            'rPTMDetermine.readers.protein_pilot_reader.ProteinPilotSearchResult': ProteinPilotSearchResult,
            'rPTMDetermine.validation_model.ValidationModel': ValidationModel,
            'pepfrag.constants.MassType': MassType,
            'pepfrag.pepfrag.ModSite': ModSite,
            'pepfrag.pepfrag.Peptide': Peptide,
            'sklearn.model_selection._search.GridSearchCV': GridSearchCV,
            'sklearn.svm._classes.LinearSVC': LinearSVC,
        }

        for expected, cls in classes.items():
            with self.subTest(classname=cls.__name__):
                self.assertEqual(expected, fullname(cls))

    def test_psm_instance(self):
        psm = PSM(None, None, Peptide('AAA', 1, []))
        self.assertEqual(
            'rPTMDetermine.peptide_spectrum_match.PSM',
            fullname(psm)
        )


class TestPacking(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self) -> None:
        self.temp_file.close()
        os.remove(self.temp_file.name)

    def _test_basic(self, value: Any, **kwargs) -> Any:
        save_to_file(value, self.temp_file.name)
        loaded = load_from_file(self.temp_file.name, **kwargs)
        self.assertEqual(value, loaded)
        return loaded

    def test_integer(self):
        self._test_basic(27)

    def test_float(self):
        self._test_basic(42.01)

    def test_complex(self):
        self._test_basic(2+1j)

    def test_string(self):
        self._test_basic('teststring')

    def test_list(self):
        # List represented as tuple due to use_list option
        self._test_basic((1, 2, 3, 4))

    def test_set(self):
        self._test_basic({2, 3, 1, 7, 6})

    def test_dict(self):
        self._test_basic({'key': 2, 'key2': (1, 2, 3)})

    def test_unsupported_type(self):
        """
        Tests that TypeError is raised on an unsupported type.

        """
        class MyClass:
            def __init__(self):
                self.a = 1

        obj = MyClass()
        save_to_file(obj, self.temp_file.name)
        with self.assertRaises(TypeError):
            load_from_file(self.temp_file.name)

    def test_pack_numpy_array(self):
        """
        Tests that a numpy array can be packed and unpacked.

        """
        arr = np.array([[1., 0.2, 0.5], [2., 0.432, 10000.]])
        save_to_file(arr, self.temp_file.name)
        loaded = load_from_file(self.temp_file.name)
        np.testing.assert_array_equal(arr, loaded)

    def test_pack_numpy_array_complex(self):
        """
        Tests that a numpy array of complex numbers can be packed and unpacked.

        """
        arr = np.array([0+1j, 1+1j])
        save_to_file(arr, self.temp_file.name)
        loaded = load_from_file(self.temp_file.name)
        np.testing.assert_array_equal(arr, loaded)

    def test_pack_numpy_array_object(self):
        """
        Tests that a numpy array of objects can be packed and unpacked.

        """
        arr = np.array([[1, 2]], dtype='object')
        save_to_file(arr, self.temp_file.name)
        loaded = load_from_file(self.temp_file.name)
        np.testing.assert_array_equal(arr, loaded)

    def test_numpy_masked_object_array(self):
        arr = np.ma.MaskedArray(
            np.array([1, 2, 3, 4, 5]),
            mask=[0, 1, 1, 0, 0],
            fill_value='?',
            dtype='object'
        )
        save_to_file(arr, self.temp_file.name)
        loaded = load_from_file(self.temp_file.name)
        np.testing.assert_array_equal(arr, loaded)

    def test_pack_enum(self):
        """
        Tests that Enums can be packed and unpacked.

        """
        mass_type = MassType.mono
        self._test_basic(mass_type)

    def test_pack_dataclass(self):
        """
        Tests that dataclasses can be packed and unpacked.

        """
        obj = ModSite(12.01, 1, 'Test')
        self._test_basic(obj)

    def test_pack_slots(self):
        """
        Tests that packing works correctly on a class using slots.

        """
        peptide = Peptide(
            'AAA', 2,
            (ModSite(12.01, 1, 'TestMod1'), ModSite(23.01232, 2, 'TestMod2'))
        )
        peptide.fragment()

        loaded = self._test_basic(peptide)
        self.assertEqual(tuple(peptide.fragment_ions), loaded.fragment_ions)

    def test_pack_dataclass_slots(self):
        """
        Tests that a dataclass using slots can be packed and unpacked.

        """
        features = Features()
        features.FracIon = 1.
        features.Charge = 2.
        features.ErrPepMass = 0.2

        self._test_basic(features)

    def test_pack_mappingproxy(self):
        """
        Tests that a mappingproxy can be packed and unpacked.

        """
        self._test_basic(MappingProxyType({'testkey': 1}))

    def test_pack_function(self):
        """
        Tests that a function can be packed and unpacked.

        """
        def my_function(a):
            return a * 2

        loaded = self._test_basic(
            my_function,
            extra_functions={
                'test_packing.my_function': my_function
            }
        )
        self.assertEqual(my_function, loaded)
        self.assertEqual(my_function(4), loaded(4))

    def test_pack_type(self):
        """
        Tests that a class itself, not an instance, can be packed.

        """
        self._test_basic(Spectrum)

    def test_pack_searchresult(self):
        """
        Tests that the SearchResult dataclass can be packed.

        """
        res = SearchResult(
            'AAABBLPK',
            (ModSite(27.9321, 1, 'testmod'),),
            1,
            '1.1.1.1.1',
            None,
            1,
            PeptideType.normal,
            1024.1
        )

        self._test_basic(res)

    def test_pack_derived_searchresult(self):
        """
        Tests that a class derived from SearchResult can be packed.

        """
        result = ProteinPilotSearchResult(
            seq='AABLPKRTYK',
            mods=(ModSite(14.09, 5, 'testmod'),),
            charge=2,
            spectrum='1.1.1.2',
            dataset=None,
            rank=1,
            pep_type=PeptideType.normal,
            theor_mz=1432.1,
            time='17.1',
            confidence=95.2,
            prec_mz=1432.2,
            itraq_peaks=None,
            proteins='Protein 1, Protein 2',
            accessions=('Accession 1', 'Accession 2'),
            itraq_ratios=None,
            background=32.1,
            used_in_quantitation=True
        )
        self._test_basic(result)

    def test_pack_psm_container(self):
        """
        Tests that the PSMContainer can be packed.

        """
        container = PSMContainer([
            PSM('testID', '1.1.1.1', Peptide('AABHNQLK', 2, ()))
        ])
        # Because we set use_list to True on unpackb
        container.data = tuple(container.data)

        self._test_basic(container)

    def test_pack_spectrum(self):
        """
        Tests that the Spectrum can be packed and unpacked.

        """
        spectrum = Spectrum(
            np.array([[101., 100.], [113., 31.], [128., 2.]]), 1302.1, 2
        )

        loaded = self._test_basic(spectrum)

        # Test this since Spectrum._peaks may be read-only
        loaded.normalize()

    def test_pack_modsite(self):
        """
        Tests that the ModSite can be packed and unpacked.

        """
        mod_site = ModSite(10.1, "nterm", "testmod")

        loaded = self._test_basic(mod_site)
        self.assertEqual(mod_site.mod, loaded.mod)

    def test_pack_validationmodel(self):
        """
        Tests that the ValidationModel can be packed and unpacked.

        """
        model = ValidationModel(
            GridSearchCV(
                LinearSVC(dual=False),
                {'C': [2 ** i for i in range(-12, 4)]},
                cv=3,
                n_jobs=1
            ),
            ['testfeature']
        )
        save_to_file(model, self.temp_file.name)
        load_from_file(self.temp_file.name)

    def test_pack_fitted_gridsearchcv(self):
        search = GridSearchCV(
            LogisticRegression(),
            {'C': [0.1, 1.]},
            cv=2
        )
        search.fit(
            np.array([
                [1.4, 2.3, 1.0],
                [2.3, 2.1, 1.1],
                [0.1, -0.2, 0.3],
                [-0.1, 0.3, 0.2]
            ]),
            np.array([1, 1, 0, 0])
        )

        save_to_file(search, self.temp_file.name)
        loaded = load_from_file(
            self.temp_file.name,
            extra_classes={'sklearn.linear_model._logistic.LogisticRegression': LogisticRegression}
        )
        self.assertEqual(
            search.predict([[1.4, 2.3, 1.0]]),
            loaded.predict([[1.4, 2.3, 1.0]])
        )

    def test_pack_rptmdetermine_function(self):
        save_to_file(write_psm_results, self.temp_file.name)
        loaded = load_from_file(
            self.temp_file.name,
            extra_functions={
                'rPTMDetermine.results.write_psm_results': write_psm_results
            }
        )
        self.assertEqual(write_psm_results, loaded)


if __name__ == '__main__':
    unittest.main()
