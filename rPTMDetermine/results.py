import csv

from . import PSMContainer, machinelearning


def write_psm_results(
        psms: PSMContainer,
        output_file: str
):
    """
    Saves the `psms` to a CSV file.

    Args:
        psms: The PSM results to save to file.
        output_file: The path to the output CSV file.

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
            'SiteDiffScore',
            'AlternativeSites',
        ])
        for psm in psms:
            writer.writerow([
                psm.data_id,
                psm.spec_id,
                psm.seq,
                psm.charge,
                ';'.join((f'{m.mod}@{m.site}' for m in psm.mods)),
                machinelearning.passes_consensus(psm.ml_scores),
                machinelearning.passes_majority(psm.ml_scores),
                psm.is_localized(),
                ';'.join(map(str, psm.ml_scores.tolist())),
                psm.site_score,
                psm.site_prob,
                psm.site_diff_score,
                ','.join([
                    ';'.join(alt[0]) + f';{alt[1]}'
                    for alt in psm.alternative_localizations
                ]) if psm.alternative_localizations is not None else ''
            ])
