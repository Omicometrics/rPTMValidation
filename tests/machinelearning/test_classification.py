import pickle
import tempfile
import unittest

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from rPTMDetermine.machinelearning.classification import Classifier


class TestClassifierPickle(unittest.TestCase):
    def test_classifier_picklable(self):
        model = Classifier(
            GridSearchCV(
                LinearSVC(),
                {'C': [2 ** i for i in range(-12, 4)]},
                cv=5
            )
        )

        with tempfile.TemporaryFile() as fh:
            pickle.dump(model, fh)
            fh.seek(0)
            pickle.load(fh)


if __name__ == '__main__':
    unittest.main()
