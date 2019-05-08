#! /usr/bin/env python3
"""
A module providing a container class for PSM objects.

"""
import collections
import csv
import os
import sys
import tqdm

import pandas as pd

import modifications
from peptide_spectrum_match import PSM

sys.path.append("../pepfrag")
from pepfrag import Peptide


class PSMContainer(collections.UserList):
    """
    """
    def __init__(self, psms=None):
        """
        """
        self.data = psms if psms is not None else []
        
    def __getitem__(self, slice):
        """
        """
        res = self.data[slice]
        return PSMContainer(res) if isinstance(res, list) else res
        
    def clean_fragment_ions(self):
        """
        Removes the cached fragment ions for the PSMs.
        
        """
        for psm in self.data:
            psm.clean_fragment_ions()
        
    def get_psms_by_seq(self, seq):
        """
        Retrieves the PSMs with the given peptide sequence.
        
        Args:
            seq (str): The peptide sequence.
            
        Returns:
            PSMContainer

        """
        return PSMContainer([p for p in self.data if p.seq == seq])
        
    def get_psms_by_id(self, data_id, spec_id):
        """
        Retrieves the PSMs with the given identifiers.
        
        Args:
            data_id (str): The data set ID.
            spec_id (str): The spectrum ID within the data set.
            
        Returns:
            PSMContainer

        """
        return PSMContainer([p for p in self.data if p.data_id == data_id and
                             p.spec_id == spec_id])
                             
    def filter_lda_score(self, threshold):
        """
        Filters the PSMs to those with an LDA score exceeding the threshold
        value.
        
        Args:
            threshold (float): The threshold LDA score to exceed.
            
        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data if p.lda_score >= threshold])
            
    def filter_lda_similarity(self, lda_threshold, sim_threshold):
        """
        Filters the PSMs to those with an LDA score exceeding the threshold
        value and a maximum similarity score exceeding the similarity score
        threshold.
        
        Args:
            lda_threshold (float): The threshold LDA score to exceed.
            sim_threshold (float): The similarity score threshold to exceed.
            
        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data if p.lda_score >= lda_threshold and
             p.max_similarity >= sim_threshold])
             
    def filter_site_prob(self, threshold):
        """
        Filters the PSMs to those without a site probability or with a site
        probability exceeding the threshold.
        
        Args:
            threshold (float): The site probability threshold.
        
        Returns:
            PSMContainer

        """
        return PSMContainer([
            p for p in self.data if p.site_prob is None or
            p.site_prob >= threshold])
             
    def ids_not_in(self, exclude_ids):
        """
        Filters the PSMs to those whose (data_id, spec_id) pair is not in the
        exclude list provided.
        
        Args:
            exclude_ids (list): A list of (data_id, spec_id) tuples.
            
        Returns:
            PSMContainer

        """
        return PSMContainer(
            [p for p in self.data
             if (p.data_id, p.spec_id) not in exclude_ids])
                             
    def get_best_psms(self):
        """
        Extracts only the PSM with the highest LDA score for each spectrum matched
        by any number of peptides.
            
        Returns:
            PSMContainer of filtered PSMs.

        """
        seen, best_psms = set(), PSMContainer()
        for psm in tqdm.tqdm(self.data):
            data_id, spec_id = psm.data_id, psm.spec_id
            comb_id = (data_id, spec_id)
            if comb_id in seen:
                continue
            seen.add(comb_id)

            max_score, max_score_psm = psm.lda_score, psm
            for other_psm in self.get_psms_by_id(data_id, spec_id):
                if other_psm.lda_score > max_score:
                    max_score, max_score_psm = other_psm.lda_score, other_psm
                
            best_psms.append(max_score_psm)
            
        return best_psms
        
    def to_df(self):
        """
        Converts the psm features, including decoy, into a pandas dataframe,
        including a flag indicating whether the features correspond to a
        target or decoy peptide and the peptide sequence.

        Returns:
            pandas.DataFrame

        """
        rows = []
        for psm in self.data:
            trow = {**{"data_id": psm.data_id, "spec_id": psm.spec_id,
                       "seq": psm.seq, "target": True}, **psm.features}
            rows.append(trow)
            if psm.decoy_id is not None:
                drow = {**{"data_id": "", "spec_id": "", "seq": psm.decoy_id.seq,
                           "target": False}, **psm.decoy_id.features}
                rows.append(drow)

        return pd.DataFrame(rows)
        
        
def read_csv(csv_file, ptmdb, spectra=None, sep="\t"):
    """
    Converts the contents of a CSV file to a PSMContainer. Each row of the
    file should contain the details of a single PSM. This function is
    designed to be used to read CSV files output by this program.
    
    Args:
        csv_file (str): The path to the CSV file.
        
    Returns:
        PSMContainer.

    """
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} does not exist - exiting")
        sys.exit(1)
        
    psms = []
    with open(csv_file, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=sep)
        for row in reader:
            psm = PSM(
                row.get("DataID", row.get("Rawset", None)),
                row["SpectrumID"],
                Peptide(
                    row["Sequence"],
                    int(row["Charge"]),
                    modifications.parse_mods(row["Modifications"], ptmdb)
                )
            )
            psm.lda_score = float(row.get("rPTMDetermineScore", None))
            site_prob = row["SiteProbability"]
            psm.site_prob = (float(site_prob) if site_prob else None)

            if "SimilarityScore" in row:
                psm.similarity_scores = [(None, None, float(row["SimilarityScore"]))]
            
            if spectra is not None:
                psm.spectrum = spectra[psm.data_id][psm.spec_id]
                
            psms.append(psm)
        
    return PSMContainer(psms)