# rPTMDetermine
rPTMDetermine provides a fully automated methodology for the validation, site 
localization and retrieval of post-translational modification (PTM) identifications 
from the database search results of tandem mass spectrometry (MS/MS) data.

# Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
    - [Configuration Options](#configuration-options)
        - [Data Set Configuration Options](#data-set-configuration-options)
    - [Sample Configuration](#sample-configuration)
- [License](#license)
- [References](#references)

# Introduction

[(Back to top)](#table-of-contents)

`rPTMDetermine` is a tool for automated validation, site localization and retrieval
of PTM identifications from protein sequence database search of tandem mass spectrometry
proteomics data. For a specified PTM, `rptmdetermine_validate.py` can be run to validate
the identifications from database search, resolving issues with the application of
global FDR control to sets of PTM identifications. This process includes site localization
and, optionally, correction of falsely-assigned deamidation.

After construction of the machine learning model in this process, the model can
be used via `rptmdetermine_retrieve.py` to retrieve PTM identifications missed 
during protein sequence database search.

`rPTMDetermine` has been most extensively tested using search results from ProteinPilot,
but further database search engines are supported, see the
[search engine](#search_engine-required) configuration option.

# Installation

[(Back to top)](#table-of-contents)

Installation is currently a manual process; we will seek to publish the package to
PyPI in the near future for easier install.

### Compatibility

`rPTMDetermine` is written using Python 3 and should be compatible with most 
operating systems. The package has been tested on
- Windows 10
- MacOS 10.15

Because `rPTMDetermine` includes C/C++ extensions, installation requires the 
presence of a C++ 11 compatible compiler on your machine.

### Instructions

1. Install Python 3 (>= version 3.6).
2. [Download](https://github.com/ikcgroup/rPTMValidation/archive/v1.0.zip) and 
unzip `rPTMDetermine` version 1.0.
3. Install our accompanying library, [`pepfrag`](https://github.com/ikcgroup/pepfrag), 
following the instructions provided.
4. Navigate to the unzipped `rPTMValidation` directory and execute 
`pip install -r requirements.txt` to install dependency packages.
5. From the `rPTMValidation` directory, execute `python setup.py install` to 
compile the C/C++ extensions and install the `rPTMDetermine` library, along with 
the scripts `rptmdetermine_validate.py` and `rptmdetermine_retrieve.py`.

# Usage

[(Back to top)](#table-of-contents)

`rPTMDetermine` ships with two scripts: `rptmdetermine_validate.py` and 
`rptmdetermine_retrieve.py`. Their behaviour is customized using a JSON 
configuration file with the options described below.

### Configuration Options

The required and optional configuration options are detailed below. Those labelled
as "Required - [SCRIPT]" are required only for the specified script.

#### `search_engine` (Required)

- Description: The database search engine used to generate the `results` files.
- Type: string, one of the following available options:
    - ProteinPilot
    - Mascot
    - Comet
    - XTandem
    - TPP
    - MSGFPlus
    - Percolator
    - PercolatorText

#### `modification` (Required)

- Description: The modification to be validated, using its Unimod name.
- Type: string.

#### `target_residues` (Required)

- Description: A list of amino acid residues targeted by the modification and 
to be validated.
- Type: array of strings (single characters).

#### `target_database` (Required)

- Description: The path to the target protein sequence database used during 
database search.
- Type: string.

#### `sim_threshold` (Required)

- Description: The threshold similarity score for validation.
- Type: number.

#### `model_file` (Required - rptmdetermine_retrieval.py)

- Description: The path to the validation `model.csv` file from 
rptmdetermine_validate.py.
- Type: string.

#### `unmod_model_file` (Required - rptmdetermine_retrieval.py)

- Description: The path to the validation `unmod_model.csv` file from 
rptmdetermine_validate.py.
- Type: string.

#### `validated_ids_file` (Required - rptmdetermine_retrieval.py)

- Description: The path to the validation results file from 
rptmdetermine_validate.py.
- Type: string.

#### `data_sets` (Required)

- Description: A dictionary/map of options for each configured data set. 
See [Data Set Configuration Options](#data-set-configuration-options) for the 
available options.
- Type: object.

#### `enzyme` (Optional)

- Description: The enzyme used to cleave the proteins during sample preparation.
- Type: string.
- Default: `"Trypsin"`. 

#### `fixed_residues` (Optional)

- Description: Fixed modifications to be applied, in the form of a dictionary/map 
of residue/terminus to modification (Unimod name). 
- Type: object.
- Default: `{}` (no fixed modifications applied).

#### `output_dir` (Optional)

- Description: The directory to which to write results files. 
- Type: string.
- Default: "`modification`_`target_residues`"

#### `correct_deamidation` (Optional)

- Description: Whether to apply `rPTMDetermine`'s deamidation correction 
algorithm to attempt to correct for non-monoisotopic precursor selection. 
- Type: boolean.
- Default: `false`.

#### `site_localization_threshold` (Optional)

- Description: The probability threshold for successful localization.
- Type: number.
- Default: `0.99`.

#### `retrieval_tolerance` (Optional)

- Description: The *m/z* tolerance for candidate matches during retrieval.
- Type: number.
- Default: `0.05`.

#### `exclude_features` (Optional)

- Description: A set of features to exclude from the validation model.
- Type: array.
- Default: `[]`.

### Data Set Configuration Options

[(Back to top)](#table-of-contents)

The `data_sets` field of the configuration file must be an object mapping unique
data set identifiers to the configurations for that data set. See 
[Sample Configuration](#sample-configuration) for examples.

#### `data_dir` (Required)

- Description: The directory within which the database search results and spectra
are located.
- Type: string.

#### `results` (Required)

- Description: The name of the database search results file, located within 
`data_dir`.
- Type: string.

#### `spectra_files` (Required)

- Description: The names of the raw mass spectra files for the `results`, located
within `data_dir`.
- Type: array.

#### `confidence` (Required - ProteinPilot)

- Description: For use with ProteinPilot search results only. The confidence score
for the desired FDR cut-off.
- Type: number.

### Sample Configuration

```json
{
    "search_engine": "ProteinPilot",
    "modification": "Nitro",
    "target_residues": [
        "Y"
    ],
    "enzyme": "Trypsin",
    "target_database": "Nitro_Y/target.fasta",
    "fixed_residues": {
        "nterm": "iTRAQ8plex",
        "K": "iTRAQ8plex",
        "C": "Carbamidomethyl"
    },
    "output_dir": "Nitro_Y",
    "correct_deamidation": true,
    "sim_threshold": 0.42,
    "model_file": "Nitro_Y/Nitro_Y_model.csv",
    "unmod_model_file": "Nitro_Y/Nitro_Y_unmod_model.csv",
    "validated_ids_file": "Nitro_Y/Nitro_Y_results.csv",
    "retrieval_tolerance": 0.05,
    "site_localization_threshold": 0.99,
    "exclude_features": [
        "Charge",
        "PepMass",
        "ErrPepMass"
    ],
    "data_sets": {
        "I19": {
            "data_dir": "RawData/I19",
            "confidence": 90.2,
            "results": "I19_PeptideSummary.txt",
            "spectra_files": [
                "I19_MGFPeaklist.mgf"
            ]
        },
        "I08": {
            "data_dir": "RawData/I08",
            "confidence": 87.7,
            "results": "I08_PeptideSummary.txt",
            "spectra_files": [
                "I08_MGFPeaklist.mgf"
            ]
        }
    }
}
```

# License

[(Back to top)](#table-of-contents)

`rPTMDetermine` is released under the [GPL-3.0](LICENSE) license.

# References

[(Back to top)](#table-of-contents)