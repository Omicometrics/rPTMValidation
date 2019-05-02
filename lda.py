#! /usr/bin/env python3
"""
A module to provide functions for the LDA (machine learning) validation
of PSMS.

"""
import bisect
import copy
import operator
from typing import List, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from peptide_spectrum_match import PSM, psms2df

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

        self.scores = {}
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
            self.scores[col] = score
        return self


def _lda_pipeline(fisher_threshold: float):
    """
    """
    return CustomPipeline(
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


def lda_model(df: pd.DataFrame, features: List[str],
              fisher_threshold: float):
    """
    Trains and returns an LDA validation model.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.

    Returns:

    """
    # TODO
    for feature in ["PepLen", "ErrPepMass", "Charge", "PepMass"]:
        if feature in features:
            features.remove(feature)

    pipeline = _lda_pipeline(fisher_threshold)

    return pipeline.fit(df[features], df["target"])


def calculate_probs(classes, preds, scores):
    """
    Calculates the normal distribution probabilities for the predictions.

    Args:
        classes (list): The possible categorization classes.
        preds (list): The class predictions.
        scores (list): The corresponding prediction scores.

    Returns:
        Dictionary mapping class to prediction probabilities.

    """
    probs = {}

    # Calculate the mean and standard deviation for each class
    stats = [(np.mean(scores[preds == cl]), np.std(scores[preds == cl]))
             for cl in classes]

    # Calculate probabilities based on the normal distribution
    for ii, _class in enumerate(classes):
        probs[int(_class)] = \
            norm.pdf((scores - stats[ii][0]) / stats[ii][1]) /\
            sum(norm.pdf((scores - mean) / std) for mean, std in stats)

    return probs


def lda_validate(df: pd.DataFrame, features: List[str],
                 fisher_threshold: float, kfold=True, **kwargs):
    """
    Trains and uses an LDA validation model using cross-validation.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.
        kfold (bool): If True, k-fold cross-validation is used.
        **kwargs: Extra keyword arguments are passed to cross_val_predict.

    Returns:

    """
    # TODO
    for feature in ["PepLen", "ErrPepMass", "Charge", "PepMass"]:
        features.remove(feature)

    X = df[features]
    y = df["target"].astype("bool")

    pipeline = _lda_pipeline(fisher_threshold)

    if kfold:
        # Perform cross validation using the custom LDA class to generate the
        # combined output for LDA.decision_function and LDA.predict.
        results = cross_val_predict(pipeline, X, y, method="decide_predict",
                                    **kwargs)
    else:
        # Train the model on the whole data set
        results = pipeline.fit(X, y).decide_predict(X)

    # Retrieve the decision_function scores and the y label predictions
    scores, preds = results[:, 0], results[:, 1]

    # Compute calibrated probabilities
    probs = calculate_probs(y.unique(), preds, scores)

    df["score"] = scores
    df["prob"] = probs[1]

    # Return the fraction of incorrect predictions and the dataframe
    # with results
    return sum(y != preds) / len(y), df, pipeline


def merge_lda_results(psms: List[PSM], lda_results) -> List[PSM]:
    """
    Merges the LDA results (score and probability) to the PSM objects.

    Args:
        psms (list of PSMs): The PSM list to filter.
        lda_results (pandas.DataFrame): The LDA validation results.

    Returns:
        PSMs with their lda_score and lda_prob attributes set.

    """
    lda_results["uid"] = lda_results.data_id + "_" + lda_results.spec_id + \
        "_" + lda_results.seq
    for psm in psms:
        idx = lda_results.index[
            (lda_results.target) & (lda_results.uid == psm.uid)].tolist()[0]
        trow = lda_results.loc[idx]
        drow = lda_results.loc[idx + 1]
        psm.lda_score, psm.lda_prob = trow.score, trow.prob
        psm.decoy_lda_score, psm.decoy_lda_prob = drow.score, drow.prob

    return psms


def apply_deamidation_correction(pipeline, psms, target_mod, proteolyzer):
    """
    Removes the deamidation modification from applicable peptide
    identifications and revalides using the trained LDA model. If the score
    for the non-deamidated analogue is greater than that for the deamidated
    peptide, the non-deamidated analogue is assigned as the peptide match for
    that spectrum.

    Args:
        pipeline (sklearn.Pipeline): The trained LDA pipeline.
        psms (list): The validated PSMs.
        target_mod (str): The modification under validation.
        proteolyzer (proteolysis.Proteolyzer)

    Returns:
        The input list of PSMs, with deamidated PSMs replaced by their
        non-deamidated counterparts if their LDA scores are higher.

        """
    for ii, psm in enumerate(psms):
        if not any(ms.mod == "Deamidated" for ms in psm.mods):
            continue
        nondeam_psm = copy.deepcopy(psm)
        nondeam_psm.peptide.mods = [ms for ms in nondeam_psm.mods
                                    if ms.mod != "Deamidated"]
        # Calculate new features
        nondeam_psm.extract_features(target_mod, proteolyzer)
        score = pipeline.decide_predict(
            psms2df([nondeam_psm]))[0, 0]
        if score > psm.lda_score:
            nondeam_psm.corrected = True

            nondeam_psm.lda_score = score

            # Reset the PSM validation attributes back to None
            for attr in ["lda_prob", "decoy_lda_score", "decoy_lda_prob"]:
                setattr(nondeam_psm, attr, None)

            psms[ii] = nondeam_psm

    return psms
    
    
def get_validation_threshold(val_results: List[Tuple[float, float]],
                             prob_threshold: float) -> float:
    """
    Finds the minimum score associated with the given probability threshold.

    Args:
        val_results (list): A list of tuples of (score, probability).
        prob_threshold (float): The probability threshold.

    Returns:
        The score threshold for the probability as a float.

    """
    val_results = sorted(val_results, key=operator.itemgetter(0))
    idx = bisect.bisect_left([r[1] for r in val_results], prob_threshold)
    return val_results[idx][0]
    
    
def get_validation_threshold_df(results_df, prob_threshold):
    """
    Finds the minimum score associated with the given probability threshold.
    
    Args:
        results_df (pandas.DataFrame): The LDA validation results DataFrame,
                                       including 'score' and 'prob' columns.
        prob_threshold (float): The probability threshold.
                                       
    Returns:
        The score threshold for the probability as a float.

    """
    return get_validation_threshold(
        list(results_df[["score", "prob"]].itertuples(
            index=False, name=None)),
        prob_threshold)
