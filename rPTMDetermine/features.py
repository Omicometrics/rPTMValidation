#! /usr/bin/env python3
"""
"""

import dataclasses
from typing import Iterator, List, Optional, Sequence, Tuple


@dataclasses.dataclass(init=False)
class Features:
    """
    A data class to represent the available features.

    """
    __slots__ = (
        "NumPeaks",
        "TotInt",
        "PepLen",
        "PepMass",
        "Charge",
        "ErrPepMass",
        "FracIon",
        "FracIonInt",
        "NumSeriesbm",
        "NumSeriesym",
        "NumIona",
        "NumIonynl",
        "NumIonbnl",
        "FracIonIntb_c1",
        "FracIonIntb_c2",
        "FracIonInty_c1",
        "FracIonInty_c2",
        "FracIon20pc",
        "NumIonb",
        "NumIony",
        "FracIonInty",
        "FracIonIntb",
        "MatchScore",
        "SeqTagm",
    )

    NumPeaks: Optional[float]
    TotInt: Optional[float]
    PepLen: Optional[float]
    PepMass: Optional[float]
    Charge: Optional[float]
    ErrPepMass: Optional[float]
    FracIon: Optional[float]
    FracIonInt: Optional[float]
    NumSeriesbm: Optional[float]
    NumSeriesym: Optional[float]
    NumIona: Optional[float]
    NumIonynl: Optional[float]
    NumIonbnl: Optional[float]
    FracIonIntb_c1: Optional[float]
    FracIonIntb_c2: Optional[float]
    FracIonInty_c1: Optional[float]
    FracIonInty_c2: Optional[float]
    FracIon20pc: Optional[float]
    NumIonb: Optional[float]
    NumIony: Optional[float]
    FracIonInty: Optional[float]
    FracIonIntb: Optional[float]
    MatchScore: Optional[float]
    SeqTagm: Optional[float]

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

    def to_list(self, features: Optional[Sequence[str]] = None) -> List[float]:
        """
        Converts the set features to a list, optionally limiting the `features`
        to be output.

        Args:
            features: The features to be output. If not provided (or None), all
                      features with non-None values will be output.

        """
        if features is None:
            features = Features.all_feature_names()

        # Python 3.8 one-liner:
        # return [v for f in features if (v := self.get(f)) is not None]
        values: List[float] = []
        for feature in features:
            val = self.get(feature)
            if val is not None:
                values.append(val)

        return values
