import csv

from . import PSMContainer, machinelearning


def write_psm_results(psms: PSMContainer, output_file: str, threshold: float):
    """
    Saves the `psms` to a CSV file.

    Args:
        psms: The PSM results to save to file.
        output_file: The path to the output CSV file.
        threshold: Score threshold for machine learning validation.

    """
    with open(output_file, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'DataID',
            'SpectrumID',
            'Sequence',
            'Charge',
            'Modifications',
            'PassesConsensus',
            'PassesMajority',
            'Localized',
            'Scores',
            'SiteScore',
            'SiteProbability',
            'SiteDiffScore'
        ])
        for psm in psms:
            writer.writerow([
                psm.data_id,
                psm.spec_id,
                psm.seq,
                psm.charge,
                ';'.join((f'{m.mod}@{m.site}' for m in psm.mods)),
                machinelearning.passes_consensus(
                    psm.ml_scores, threshold
                ),
                machinelearning.passes_majority(
                    psm.ml_scores, threshold
                ),
                psm.is_localized(),
                ';'.join(map(str, psm.ml_scores.tolist())),
                psm.site_score,
                psm.site_prob,
                psm.site_diff_score
            ])
