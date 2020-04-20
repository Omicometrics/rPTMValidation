#! /usr/bin/env python3
"""
"""

import dataclasses
from typing import Iterator, List, Optional, Tuple


@dataclasses.dataclass(init=False)
class Features:
    """
    A data class to represent the available features.

    """
    __slots__ = (
        "PepLen",
        "PepMass",
        "Charge",
        "ErrPepMass",
        "TotalIntMod",
        "FracIon",
        "FracIonInt",
        "FracIon20pc",
        "NumIonb",
        "NumIony",
        "NumIonb2l",
        "NumIony2l",
        "MatchScore",
        "MatchScoreMod",
        "SeqTagm",
        "MissedCleavages",)

    PepLen: Optional[float]
    PepMass: Optional[float]
    Charge: Optional[float]
    ErrPepMass: Optional[float]
    TotalIntMod: Optional[float]
    FracIon: Optional[float]
    FracIonInt: Optional[float]
    FracIon20pc: Optional[float]
    NumIonb: Optional[float]
    NumIony: Optional[float]
    NumIonb2l: Optional[float]
    NumIony2l: Optional[float]
    MatchScore: Optional[float]
    MatchScoreMod: Optional[float]
    SeqTagm: Optional[float]
    MissedCleavages: Optional[float]

    @staticmethod
    def all_feature_names() -> Tuple[str, ...]:
        """
        Extracts the available feature names.

        Returns:
            Tuple of feature names.

        """
        return Features.__slots__

    def __init__(self):
        """
        Initialize the data class by defaulting all features to None.
        """
        for feature in Features.__slots__:
            setattr(self, feature, None)

    def __iter__(self) -> Iterator[Tuple[str, Optional[float]]]:
        """
        Iterates through the features to retrieve the corresponding values.
        Returns:
            Iterator of (feature, value) tuples.
        """
        for feature in Features.all_feature_names():
            val = getattr(self, feature)
            if val is not None:
                yield feature, val

    def feature_names(self) -> List[str]:
        """
        Returns the names of the initialized features.

        Returns:
            List of initialized feature names.

        """
        return [name for name, val in self]

    def get(self, feature: str) -> Optional[float]:
        """
        Returns the value of the given feature.

        """
        return getattr(self, feature)

    def set(self, feature: str, value: float):
        """
        Sets the value of the given feature.

        """
        setattr(self, feature, value)

    def isvalid(self):
        """
        Check whether the feature is a valid feature
        set by identifying NoneType

        """
        return not any(getattr(self, feature) is None
                       for feature in Features.__slots__)

