#! /usr/bin/env python3
"""
A module to provide functions for the LDA (machine learning) validation
of PSMS.

"""
import itertools
from typing import List
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Silence this since it arises when converting ints to float in StandardScaler
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class CustomPipeline(Pipeline):
    """
    A simple subclass of sklearn's Pipeline to allow cross_val_predict to
    return the combined results of multiple base class methods.

    """
    def decide_predict(self, X):
        """
        Calls the decision_function and predict methods of the base class.

        """
        return np.transpose([
            self.decision_function(X),
            self.predict(X)
        ])


class FisherScoreSelector():
    """
    A custom feature selector for use with sklearn.Pipeline. This selector
    will extract the features whose Fisher scores exceed the specified
    threshold.

    """
    def __init__(self, threshold):
        """
        Initialize the selector with the desired score threshold.

        Args:
            threshold (float): The minimum Fisher score to retain features.

        """
        self.threshold = threshold

        self._features = []

    def transform(self, X):
        """
        Transform the input by extracting the data for the features whose
        Fisher scores exceeded the threshold.

        """
        return X[self._features]

    def fit(self, X, y):
        """
        Fit the selector by calculating the Fisher scores for each feature
        and storing those which exceed the threshold.

        """
        for col in X.columns:
            vals1 = X[col].loc[y]
            vals2 = X[col].loc[~y]
            score = ((vals1.mean() - vals2.mean()) ** 2) / \
                    (vals1.std() ** 2 + vals2.std() ** 2)
            if score > self.threshold:
                self._features.append(col)
        return self


'''def ldaperm(Xf, Xrandf, sampleix):
    """
    Linear discriminant analysis
    """
    medianxv = np.median(np.array(Xf)[sampleix], axis=0)
    medianxv[medianxv == 0] = 1.
    Xm = Xf / medianxv
    Xn = Xrandf / medianxv
    y = np.concatenate((np.ones(sum(sampleix)), np.zeros(sum(sampleix))))
    X = np.concatenate((Xm[sampleix], Xn[sampleix]))
    err, ys, probs, _ = lda_validate(X, y, 10)
    model = LDA().fit(X, y)
    print(err, sum(probs[1][:sum(sampleix)] >= 0.99),
          sum(probs[1][sum(sampleix):] >= 0.99))
    return ys, probs[1], model, medianxv'''


def lda_validate(df: pd.DataFrame, features: List[str],
                 fisher_threshold: float, **kwargs):
    """
    Trains and uses an LDA validation model using cross-validation.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.
        **kwargs: Extra keyword arguments are passed to cross_val_predict.

    Returns:

    """
    X = df[features]
    y = df["target"]

    pipeline = CustomPipeline(
        [
            # Fisher score selection of features is included in the piepline
            # since it should be performed independently on each cross
            # validation fold
            ("fisher_selection", FisherScoreSelector(fisher_threshold)),
            # Scale the features to have mean 0 and unit variance
            ("scaler", StandardScaler()),
            # Perform LDA
            ("lda", LDA())
        ])

    # Perform cross validation using the custom LDA class to generate the
    # combined output for LDA.decision_function and LDA.predict.
    results = cross_val_predict(pipeline, X, y, method="decide_predict",
                                **kwargs)

    # Retrieve the decision_function scores and the y label predictions
    scores, preds = results[:, 0], results[:, 1]

    # Compute calibrated probabilities
    probs = {}
    # Calculate the mean and standard deviation for each class
    stats = [(np.mean(scores[preds == cl]), np.std(scores[preds == cl]))
             for cl in y.unique()]
    # Calculate probabilities based on the normal distribution
    for ii, _class in enumerate(y.unique()):
        probs[int(_class)] = \
            norm.pdf((scores - stats[ii][0]) / stats[ii][1]) /\
            sum(norm.pdf((scores - mean) / std) for mean, std in stats)

    #bprob = probs[1] / sum(itertools.chain(*probs.values()))
    
    df["score"] = scores
    df["prob"] = probs[1]

    # Return the fraction of incorrect predictions
    #return sum(y != preds) / len(y), scores, probs#, bprob
    return sum(y != preds) / len(y), df
