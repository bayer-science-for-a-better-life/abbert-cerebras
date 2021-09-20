abbert2: Language Models for Antibodies
=======================================

Welcome to abbert2 :wave:, a library to enable training of machine learning
models over large repertoires of antibodies.

**abbert2 is young and under heavy development, so expect rough edges**

Getting started
---------------

```shell
# clone the repo
git clone https://github.com/bayer-science-for-a-better-life/abbert-cerebras.git abbert2
cd abbert2

# create the conda environment
conda create -f environment.yml

# activate it
conda activate abbert2

# if everything worked well, this should work
# it is a small example of how to access the preprocessed data dumps from OAS
python examples/oas101.py
```

<details>
<summary>Click to see the expected output!</summary>
```
UNIT: ('paired', 'Setliff_2019', 'SRR10313335_paired')
{'age': 'no',
 'bsource': 'PBMC',
 'btype': 'Unsorted-B-Cells',
 'chain': 'Paired',
 'disease': 'HIV',
 'download_date': Timestamp('2021-09-19 03:56:33+0000', tz='UTC'),
 'download_error': None,
 'has_original_csv': False,
 'heavy_cdr3_max_length': 25,
 'isotype': 'All',
 'longitudinal': 'Year-23',
 'needs_redownload': True,
 'oas_subset': 'paired',
 'online_csv_size_bytes': 1510324,
 'online_modified_date': Timestamp('2021-07-31 13:44:00+0000', tz='UTC'),
 'original_local_csv_mdate': None,
 'original_url': 'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Setliff_2019/csv/SRR10313335_paired.csv.gz',
 'publication_link': 'https://doi.org/10.1016/j.cell.2019.11.003',
 'sequences_file_size': 73103,
 'sequences_miss_processing': True,
 'sequences_num_records': 100,
 'sequencing_run': 'SRR10313335',
 'species': 'human',
 'study_author': 'Setliff et al., 2019',
 'study_id': 'Setliff_2019',
 'subject': 'Donor-N90',
 'theoretical_num_sequences_total': None,
 'theoretical_num_sequences_unique': 1444,
 'unit_id': 'SRR10313335_paired',
 'vaccine': 'None'}
Unit disease=HIV, vaccine=None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 100 entries, 1399 to 1190
Data columns (total 76 columns):
 #   Column                                   Non-Null Count  Dtype  
---  ------                                   --------------  -----  
 0   locus_heavy                              100 non-null    object 
 1   locus_light                              100 non-null    object 
 2   stop_codon_heavy                         100 non-null    object 
 3   stop_codon_light                         100 non-null    object 
 4   vj_in_frame_heavy                        100 non-null    object 
 5   vj_in_frame_light                        100 non-null    object 
 6   productive_heavy                         100 non-null    object 
 7   productive_light                         100 non-null    object 
 8   rev_comp_heavy                           100 non-null    object 
 9   rev_comp_light                           100 non-null    object 
 10  v_call_heavy                             100 non-null    object 
 11  v_call_light                             100 non-null    object 
 12  d_call_heavy                             99 non-null     object 
 13  d_call_light                             0 non-null      float64
 14  j_call_heavy                             100 non-null    object 
 15  j_call_light                             100 non-null    object 
 16  junction_aa_heavy                        100 non-null    object 
 17  junction_aa_light                        100 non-null    object 
 18  junction_aa_length_heavy                 100 non-null    UInt16 
 19  junction_aa_length_light                 100 non-null    UInt16 
 20  anarci_status_heavy                      100 non-null    object 
 21  anarci_status_light                      100 non-null    object 
 22  unfit_heavy                              100 non-null    bool   
 23  has_unexpected_insertions_heavy          100 non-null    bool   
 24  has_mutated_conserved_cysteines_heavy    100 non-null    bool   
 25  has_wrong_sequence_reconstruction_heavy  0 non-null      object 
 26  has_wrong_cdr3_reconstruction_heavy      100 non-null    bool   
 27  has_kappa_gap_21_heavy                   100 non-null    bool   
 28  has_long_cdr1_heavy                      100 non-null    bool   
 29  has_long_cdr2_heavy                      100 non-null    bool   
 30  has_long_cdr3_heavy                      100 non-null    bool   
 31  has_insertions_heavy                     100 non-null    bool   
 32  fw1_start_heavy                          100 non-null    int64  
 33  fw1_length_heavy                         100 non-null    int64  
 34  cdr1_start_heavy                         100 non-null    int64  
 35  cdr1_length_heavy                        100 non-null    int64  
 36  fw2_start_heavy                          100 non-null    int64  
 37  fw2_length_heavy                         100 non-null    int64  
 38  cdr2_start_heavy                         100 non-null    int64  
 39  fw3_start_heavy                          100 non-null    int64  
 40  fw3_length_heavy                         100 non-null    int64  
 41  cdr2_length_heavy                        100 non-null    int64  
 42  cdr3_start_heavy                         100 non-null    int64  
 43  cdr3_length_heavy                        100 non-null    int64  
 44  fw4_start_heavy                          100 non-null    int64  
 45  fw4_length_heavy                         100 non-null    int64  
 46  aligned_sequence_heavy                   100 non-null    object 
 47  positions_heavy                          100 non-null    object 
 48  insertions_heavy                         80 non-null     object 
 49  unfit_light                              100 non-null    bool   
 50  has_unexpected_insertions_light          100 non-null    bool   
 51  has_mutated_conserved_cysteines_light    100 non-null    bool   
 52  has_wrong_sequence_reconstruction_light  0 non-null      object 
 53  has_wrong_cdr3_reconstruction_light      100 non-null    bool   
 54  has_kappa_gap_21_light                   100 non-null    bool   
 55  has_long_cdr1_light                      100 non-null    bool   
 56  has_long_cdr2_light                      100 non-null    bool   
 57  has_long_cdr3_light                      100 non-null    bool   
 58  has_insertions_light                     100 non-null    bool   
 59  fw1_start_light                          100 non-null    int64  
 60  fw1_length_light                         100 non-null    int64  
 61  cdr1_start_light                         100 non-null    int64  
 62  cdr1_length_light                        100 non-null    int64  
 63  fw2_start_light                          100 non-null    int64  
 64  fw2_length_light                         100 non-null    int64  
 65  cdr2_start_light                         100 non-null    int64  
 66  fw3_start_light                          100 non-null    int64  
 67  fw3_length_light                         100 non-null    int64  
 68  cdr2_length_light                        100 non-null    int64  
 69  cdr3_start_light                         100 non-null    int64  
 70  cdr3_length_light                        100 non-null    int64  
 71  fw4_start_light                          100 non-null    int64  
 72  fw4_length_light                         100 non-null    int64  
 73  aligned_sequence_light                   100 non-null    object 
 74  positions_light                          100 non-null    object 
 75  insertions_light                         0 non-null      object 
dtypes: UInt16(2), bool(18), float64(1), int64(28), object(27)
memory usage: 46.9+ KB
An antibody light chain: [b'E' b'I' b'V' b'M' b'T' b'Q' b'S' b'P' b'A' b'T' b'L' b'S' b'V' b'S'
 b'P' b'G' b'E' b'R' b'A' b'T' b'L' b'S' b'C' b'R' b'A' b'S' b'Q' b'S'
 b'V' b'S' b'S' b'N' b'L' b'A' b'W' b'Y' b'Q' b'Q' b'K' b'P' b'G' b'Q'
 b'A' b'P' b'R' b'L' b'L' b'I' b'Y' b'G' b'A' b'S' b'T' b'R' b'A' b'T'
 b'G' b'I' b'P' b'A' b'R' b'F' b'S' b'G' b'S' b'G' b'S' b'G' b'T' b'E'
 b'F' b'T' b'L' b'T' b'I' b'S' b'S' b'L' b'Q' b'S' b'E' b'D' b'F' b'A'
 b'V' b'Y' b'Y' b'C' b'Q' b'Q' b'Y' b'N' b'N' b'W' b'P' b'P' b'L' b'T'
 b'F' b'G' b'G' b'G' b'T' b'K' b'V' b'E' b'I' b'K']
An antibody heavy chain: [b'E' b'V' b'Q' b'L' b'V' b'E' b'S' b'G' b'G' b'G' b'L' b'V' b'K' b'P'
 b'G' b'G' b'S' b'L' b'R' b'L' b'S' b'C' b'A' b'A' b'S' b'A' b'F' b'T'
 b'F' b'S' b'N' b'A' b'W' b'M' b'S' b'W' b'V' b'R' b'Q' b'A' b'P' b'G'
 b'K' b'G' b'L' b'E' b'W' b'V' b'G' b'L' b'I' b'K' b'S' b'K' b'S' b'D'
 b'G' b'G' b'T' b'A' b'D' b'Y' b'A' b'A' b'P' b'V' b'K' b'G' b'R' b'F'
 b'T' b'I' b'S' b'R' b'D' b'D' b'S' b'K' b'N' b'T' b'L' b'F' b'L' b'Q'
 b'M' b'N' b'S' b'L' b'K' b'S' b'E' b'D' b'T' b'A' b'V' b'Y' b'Y' b'C'
 b'T' b'T' b'R' b'L' b'D' b'Y' b'S' b'D' b'Y' b'F' b'F' b'Y' b'H' b'Y'
 b'Y' b'F' b'M' b'D' b'V' b'W' b'G' b'K' b'G' b'T' b'T' b'V' b'T' b'V'
 b'S' b'S']
--------------------------------------------------------------------------------
UNIT: ('unpaired', 'Bhiman_2015', 'SRR2126755_Heavy_IGHM')
{'age': 'no',
 'bsource': 'PBMC',
 'btype': 'Unsorted-B-Cells',
 'chain': 'Heavy',
 'disease': 'HIV',
 'download_date': Timestamp('2021-09-19 03:56:43+0000', tz='UTC'),
 'download_error': None,
 'has_original_csv': False,
 'heavy_cdr3_max_length': 16,
 'isotype': 'IGHM',
 'longitudinal': 'no',
 'needs_redownload': True,
 'oas_subset': 'unpaired',
 'online_csv_size_bytes': 2469,
 'online_modified_date': Timestamp('2021-08-09 16:10:00+0000', tz='UTC'),
 'original_local_csv_mdate': None,
 'original_url': 'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Bhiman_2015/csv/SRR2126755_Heavy_IGHM.csv.gz',
 'publication_link': 'https://doi.org/10.1038/nm.3963',
 'sequences_file_size': 31089,
 'sequences_miss_processing': False,
 'sequences_num_records': 2,
 'sequencing_run': 'SRR2126755',
 'species': 'human',
 'study_author': 'Bhiman et al., 2015',
 'study_id': 'Bhiman_2015',
 'subject': 'CAP256',
 'theoretical_num_sequences_total': 2,
 'theoretical_num_sequences_unique': 2,
 'unit_id': 'SRR2126755_Heavy_IGHM',
 'vaccine': 'None'}
Unit disease=HIV, vaccine=None
<class 'pandas.core.frame.DataFrame'>
Int64Index: 2 entries, 0 to 1
Data columns (total 41 columns):
 #   Column                                   Non-Null Count  Dtype 
---  ------                                   --------------  ----- 
 0   locus_heavy                              2 non-null      object
 1   stop_codon_heavy                         2 non-null      object
 2   vj_in_frame_heavy                        2 non-null      object
 3   v_frameshift_heavy                       2 non-null      object
 4   productive_heavy                         2 non-null      object
 5   rev_comp_heavy                           2 non-null      object
 6   complete_vdj_heavy                       2 non-null      object
 7   v_call_heavy                             2 non-null      object
 8   d_call_heavy                             2 non-null      object
 9   j_call_heavy                             2 non-null      object
 10  junction_aa_heavy                        2 non-null      object
 11  junction_aa_length_heavy                 2 non-null      UInt16
 12  redundancy_heavy                         2 non-null      int64 
 13  anarci_status_heavy                      2 non-null      object
 14  unfit_heavy                              2 non-null      bool  
 15  has_unexpected_insertions_heavy          2 non-null      bool  
 16  has_mutated_conserved_cysteines_heavy    2 non-null      bool  
 17  has_wrong_sequence_reconstruction_heavy  0 non-null      object
 18  has_wrong_cdr3_reconstruction_heavy      2 non-null      bool  
 19  has_kappa_gap_21_heavy                   2 non-null      bool  
 20  has_long_cdr1_heavy                      2 non-null      bool  
 21  has_long_cdr2_heavy                      2 non-null      bool  
 22  has_long_cdr3_heavy                      2 non-null      bool  
 23  has_insertions_heavy                     2 non-null      bool  
 24  fw1_start_heavy                          2 non-null      int64 
 25  fw1_length_heavy                         2 non-null      int64 
 26  cdr1_start_heavy                         2 non-null      int64 
 27  cdr1_length_heavy                        2 non-null      int64 
 28  fw2_start_heavy                          2 non-null      int64 
 29  fw2_length_heavy                         2 non-null      int64 
 30  cdr2_start_heavy                         2 non-null      int64 
 31  fw3_start_heavy                          2 non-null      int64 
 32  fw3_length_heavy                         2 non-null      int64 
 33  cdr2_length_heavy                        2 non-null      int64 
 34  cdr3_start_heavy                         2 non-null      int64 
 35  cdr3_length_heavy                        2 non-null      int64 
 36  fw4_start_heavy                          2 non-null      int64 
 37  fw4_length_heavy                         2 non-null      int64 
 38  aligned_sequence_heavy                   2 non-null      object
 39  positions_heavy                          2 non-null      object
 40  insertions_heavy                         2 non-null      object
dtypes: UInt16(1), bool(9), int64(15), object(16)
memory usage: 536.0+ bytes
An antibody heavy chain: [b'E' b'V' b'Q' b'L' b'V' b'E' b'S' b'G' b'G' b'G' b'L' b'V' b'Q' b'P'
 b'G' b'G' b'S' b'L' b'R' b'L' b'S' b'C' b'A' b'A' b'S' b'G' b'F' b'T'
 b'F' b'S' b'T' b'Y' b'D' b'M' b'H' b'W' b'V' b'R' b'Q' b'G' b'A' b'G'
 b'K' b'G' b'P' b'E' b'W' b'V' b'A' b'G' b'I' b'G' b'R' b'A' b'G' b'D'
 b'T' b'Y' b'Y' b'P' b'G' b'S' b'E' b'K' b'G' b'R' b'F' b'T' b'I' b'S'
 b'R' b'E' b'N' b'A' b'K' b'N' b'S' b'L' b'Y' b'L' b'E' b'M' b'N' b'S'
 b'L' b'R' b'A' b'G' b'D' b'T' b'A' b'V' b'Y' b'Y' b'C' b'A' b'R' b'R'
 b'R' b'Q' b'G' b'S' b'A' b'S' b'Y' b'S' b'D' b'A' b'F' b'D' b'I' b'W'
 b'G' b'Q' b'G' b'T' b'M' b'V' b'T' b'V' b'S' b'S']
--------------------------------------------------------------------------------
```
</details>


OAS
---

The [Observed Antibody Space](http://opig.stats.ox.ac.uk/webapps/oas/) (OAS) data
is licensed under a [CC-BY 4.0 license](https://creativecommons.org/licences/by/4.0/).
