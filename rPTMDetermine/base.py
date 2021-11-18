import collections


# similarity scores
SimilarityScore = collections.namedtuple("SimilarityScore",
                                         ["uid", "score", "analogue"])

# ion annotations
Annotation = collections.namedtuple(
    "Annotation", ["ion", "mz", "peak_intensity", "mz_diff"]
)

# Localization info: details
InfoFields = ["target", "model_residue", "top_sites", "next_sites",
              "top_score", "next_score", "diff", "is_loc"]
LocInfo = collections.namedtuple("LocInfo", InfoFields,
                                 defaults=(None,) * len(InfoFields))

# Localization info: summary from different models
ModLocFields = ["modification", "loc_residues", "top_score",
                "alternative_score", "score_difference", "sites",
                "alternative_sites", "site_difference", "is_localized",
                "frac_supports"]
ModLocates = collections.namedtuple("ModLocates", ModLocFields,
                                    defaults=(None,) * len(ModLocFields))