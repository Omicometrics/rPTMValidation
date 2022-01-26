"""
This module provides basic objects for other modules.

"""

import collections
import operator

from typing import Dict, Callable
from readers import SearchEngine, SearchResult

# score getter from different search engines
ScoreGetter = Callable[[SearchResult], float]
ScoreGetterMap: Dict[SearchEngine, ScoreGetter] = {
    SearchEngine.Mascot: operator.attrgetter("ionscore"),
    SearchEngine.ProteinPilot: operator.attrgetter("confidence"),
    SearchEngine.ProteinPilotXML: operator.attrgetter("confidence"),
    SearchEngine.TPP: operator.attrgetter("pprophet_prob"),
    SearchEngine.Percolator: operator.attrgetter("q_value"),
    SearchEngine.Comet: lambda r: r.scores["xcorr"],
    SearchEngine.XTandem: lambda r: r.scores["hyperscore"],
    SearchEngine.MSFragger: lambda r: r.scores["hyperscore"]
}


# similarity scores
SimilarityScore = collections.namedtuple(
    "SimilarityScore", ["uid", "score", "analogue"])

# ion annotations
Annotation = collections.namedtuple(
    "Annotation", ["ion", "mz", "peak_intensity", "mz_diff"])

# Localization info: details
InfoFields = ["target", "model_residue", "top_sites", "next_sites",
              "top_score", "next_score", "diff", "is_loc"]
LocInfo = collections.namedtuple(
    "LocInfo", InfoFields, defaults=(None,) * len(InfoFields))

# Localization info: summary from different models
ModLocFields = ["modification", "loc_residues", "top_score",
                "alternative_score", "score_difference", "sites",
                "alternative_sites", "site_difference", "is_localized",
                "frac_supports"]
ModLocates = collections.namedtuple(
    "ModLocates", ModLocFields, defaults=(None,) * len(ModLocFields))
