#! /usr/bin/env python3
"""
A module to provide functions for the LDA (machine learning) validation
of PSMS.

"""
import copy
from typing import List
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from peptide_spectrum_match import PSM, psms2df

# Silence this since it arises when converting ints to float in StandardScaler
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class CustomPipeline(Pipeline):
    """
    A simple subclass of sklearn's Pipeline to provide a convenience function
    to return the combined results of multiple base class methods.

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
            norm.pdf(scores, stats[ii][0], stats[ii][1]) /\
            sum(norm.pdf(scores, mean, std) for mean, std in stats)

    return probs
    
    
def calculate_threshold(target_scores, decoy_scores, prob=0.99):
    """
    Solves for the score threshold corresponding to the given probability.
    
    Args:
        target_scores (np.ndarray)
        decoy_scores (np.ndarray)
        prob (float)
        
    Returns:
        float

    """
    m_t, s_t = target_scores.mean(), target_scores.std()
    m_d, s_d = decoy_scores.mean(), decoy_scores.std()

    a = - 2 * s_d * s_d + 2 * s_t * s_t 
    b = 4 * m_t * s_d * s_d - 4 * m_d * s_t * s_t
    c = (- 2 * s_d * s_d * m_t * m_t + 2 * s_t * s_t * m_d * m_d -
         (4 * s_d * s_d * s_t * s_t) *
            (np.log(prob / s_d) - np.log((1 - prob) / s_t)))
    
    x1 = (-b + np.sqrt((b * b) - (4 * a * c))) / (2 * a)
    x2 = (-b - np.sqrt((b * b) - (4 * a * c))) / (2 * a)
    
    return max((x1, x2))

    
def _lda_validate(df: pd.DataFrame, features: List[str],
                  fisher_threshold: float, full_lda_threshold,
                  cv=10):
    """
    Trains and uses an LDA validation model using cross-validation.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.
        cv (int, optional): If None, no cross validation will be used.
                            Otherwise, this should be an integer for the
                            number of folds.

    Returns:

    """
    # TODO: remove
    features = [f for f in features
                if f not in ["PepLen", "ErrPepMass", "Charge", "PepMass"]]

    X = df[features]
    y = df["target"].astype("bool")

    pipeline = _lda_pipeline(fisher_threshold)

    if cv is None:
        # Train the model on the whole data set
        model = pipeline.fit(X, y)
        scores = model.decision_function(X)
        preds = model.predict(X)
        probs = calculate_probs(y.unique(), preds, scores)[1]
        full_lda_threshold = calculate_threshold(scores[preds == 1],
                                                 scores[preds == 0])
    else:
        skf = StratifiedKFold(n_splits=cv)
        results = np.zeros((len(X), 3))
        for train_idx, test_idx in skf.split(X, y):
            model = pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            X_test = X.iloc[test_idx]
            scores = model.decision_function(X_test)
            preds = model.predict(X_test)

            lda_threshold = calculate_threshold(scores[preds == 1],
                                                scores[preds == 0])
            scores += full_lda_threshold - lda_threshold
            results[test_idx, 0], results[test_idx, 1] = scores, preds
            results[test_idx, 2] = calculate_probs(y.unique(), preds, scores)[1]

        scores, preds, probs = results[:, 0], results[:, 1], results[:, 2]

    df["score"], df["prob"] = scores, probs

    return sum(y != preds) / len(y), df, pipeline, full_lda_threshold
    
    
def lda_validate(df: pd.DataFrame, features: List[str],
                 fisher_threshold: float, full_lda_threshold,
                 cv=10):
    """
    Trains and uses an LDA validation model using cross-validation.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.
        cv (int, optional): If None, no cross validation will be used.
                            Otherwise, this should be an integer for the
                            number of folds.

    Returns:

    """
    _, results, _, _ = _lda_validate(
            df, features, fisher_threshold, full_lda_threshold, cv=cv)
    return results
    
    
def lda_model(df: pd.DataFrame, features: List[str],
              fisher_threshold: float):
    """
    Trains and uses an LDA validation model.

    Args:
        df (pandas.DataFrame): The features and target labels for the PSMs.
        features (list): The names of the feature columns.
        fisher_threshold (float): The minimum required Fisher score for
                                  feature selection.

    Returns:

    """
    _, _, model, lda_threshold = _lda_validate(df, features, fisher_threshold,
                                              None, cv=None)
    return model, lda_threshold


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
            # TODO: calculate the new probability?

            # Reset the PSM validation attributes back to None
            for attr in ["lda_prob", "decoy_lda_score", "decoy_lda_prob"]:
                setattr(nondeam_psm, attr, None)

            psms[ii] = nondeam_psm

    return psms
