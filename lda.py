#! /usr/bin/env python3
"""
A module to provide functions for the LDA (machine learning) validation
of PSMS.

"""
import itertools
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict


class LDA(LinearDiscriminantAnalysis):
    """
    A simple subclass of sklearn's LinearDiscriminantAnalysis to allow
    cross_val_predict to return the combined results of multiple base class
    methods.

    """
    def decide_predict(self, X):
        """
        Calls the decision_function and predict methods of the base class.

        """
        return np.transpose([
            self.decision_function(X),
            self.predict(X)
        ])


def calc_fisher_scores(df: pd.DataFrame, features: List[str],
                       class_col: str) -> Dict[str, float]:
    """
    Calculates the Fisher scores the features in the DataFrame.

    Args:
        df (pandas.DataFrame): The feature DataFrame.
        features (list): The list of feature names.
        class_col (str): The name of the column corresponding to the class
                         label.

    Returns:
        dict: A map of feature name to Fisher score.

    """
    scores = {}
    for feature in features:
        vals1 = df.loc[df[class_col], df[feature]]
        vals2 = df.loc[df[~class_col], df[feature]]
        scores[feature] = ((vals1.mean() - vals2.mean) ** 2) / \
                          (vals1.std() ** 2 + vals2.std() ** 2)
    return scores


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
    # Calculate the Fisher scores of the features
    fisher_scores = calc_fisher_scores(df, features, "target")

    # Split out the machine learning features and the target/decoy label,
    # retaining only those features with Fisher scores exceeding the threshold
    X = df[[f for f in features if fisher_scores[f] > fisher_threshold]]
    y = df["target"]

    # Perform cross validation using the custom LDA class to generate the
    # combined output for LDA.decision_function and LDA.predict.
    results = cross_val_predict(LDA(), X, y, method="decide_predict",
                                **kwargs)

    # Retrieve the decision_function scores and the y label predictions
    scores, preds = results[:, 0], results[:, 1]

    # Find the number of incorrect predicts
    nerr = sum(y != preds)

    # Compute calibrated probabilities
    probs = {}
    # Calculate the mean and standard deviation for each class
    stats = [(np.mean(scores[preds == cl]), np.std(scores[preds == cl]))
             for cl in y.unique()]
    # Calculate probabilities based on the normal distribution
    for ii, cl in enumerate(y.unique()):
        probs[int(cl)] = norm.pdf((scores - stats[ii][0]) / stats[ii][1]) /\
            sum(norm.pdf((scores - mean) / std) for mean, std in stats)

    bprob = probs[1] / sum(itertools.chain(*probs.values()))

    return nerr / len(y), scores, probs, bprob
