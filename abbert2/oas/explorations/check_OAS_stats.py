# script to load OAS and retrieve statistics for various properties / issues

# TODO: later, when available, compute stats for maturity related properties e.g. identity to germline
#   maybe look at the raw ANARCI status too ?

########################################################################################################################
# example of unit nice_metadata
# {'oas_subset': 'unpaired', 'study_id': 'Galson_2015', 'unit_id': 'SRR3990830_Heavy_Bulk', 'download_date':
# Timestamp('2021-11-19 20:44:52+0000', tz='UTC'), 'online_modified_date': Timestamp('2021-07-30 15:53:00+0000', tz='UTC'),
# 'online_csv_size_bytes': 259155, 'sequencing_run': 'SRR3990830', 'publication_link': 'https://doi.org/10.1038/icb.2015.57',
# 'study_author': 'Galson_2015 et al., 2015', 'study_year': 2015, 'species': 'human', 'age': '30-70', 'bsource': 'PBMC',
# 'btype': 'Plasma-B-Cells', 'subject': 'Subject-1009', 'disease': 'None', 'vaccine': 'MenACWY-conjugate', 'longitudinal': 'Visit-2',
# 'chain': 'Heavy', 'isotype': 'Bulk', 'theoretical_num_sequences_unique': 731, 'theoretical_num_sequences_total': 837,
# 'original_url': 'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Galson_2015/csv/SRR3990830_Heavy_Bulk.csv.gz',
# 'has_original_csv': False, 'original_local_csv_mdate': None, 'download_error': None, 'needs_redownload': True,
# 'sequences_file_size': 31051, 'has_broken_sequences_file': False, 'sequences_num_records': 11, 'sequences_miss_processing': True,
# 'num_heavy_sequences': 11, 'num_light_sequences': 0, 'heavy_cdr3_max_length': 22}

########################################################################################################################
# raw df columns
# 'index_in_unit', 'chain', 'locus', 'v_call', 'd_call', 'j_call',
# 'sequence_aa', 'imgt_positions', 'imgt_insertions', 'rev_comp',
# 'junction_aa', 'junction_aa_length', 'fwr1_start', 'fwr1_length',
# 'cdr1_start', 'cdr1_length', 'fwr2_start', 'fwr2_length', 'cdr2_start',
# 'fwr3_start', 'fwr3_length', 'cdr2_length', 'cdr3_start', 'cdr3_length',
# 'fwr4_start', 'fwr4_length', 'redundancy', 'stop_codon', 'vj_in_frame',
# 'v_frameshift', 'productive', 'complete_vdj', 'has_insertions',
# 'has_unexpected_insertions', 'has_mutated_conserved_cysteines',
# 'has_wrong_cdr3_reconstruction', 'has_kappa_gap_21', 'anarci_deletions',
# 'anarci_insertions', 'anarci_missing_conserved_cysteine',
# 'anarci_unusual_residue', 'anarci_fwr1_shorter_than_imgt_defined',
# 'anarci_fwr4_shorter_than_imgt_defined',
# 'anarci_cdr3_is_over_37_aa_long'

########################################################################################################################
# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html#Overview = region lengths in IMGT num.
# --> what properties we want to check = which df columns
# full sequence replicas (e.g. >3) = sequence_aa / (redundancy but it is within the same study cf. filtering.py?)
# CDR3 replicas = sequence_aa[cdr3_start: cdr3_start+cdr3_length]
# too long region lengths (e.g. CDR3 length cutoff of 37) = {region}_length / anarci_cdr3_is_over_37_aa_long
# unusual insertions/deletions = has_insertions / has_unexpected_insertions / anarci_insertions / imgt_insertions / anarci_deletions
# truncated beginning (FW1>=20) or ending (FW4>=10) = anarci_fwr1_shorter_than_imgt_defined / anarci_fwr4_shorter_than_imgt_defined
# missing regions = complete_vdj / has_wrong_cdr3_reconstruction
# bulk assignment (e.g. too low isotype assignment, short or missing CH1) = isotype (from metadata)
# productive sequence = productive
# lack of conserved cysteines = anarci_missing_conserved_cysteine / has_mutated_conserved_cysteines
# unusual residues = anarci_unusual_residue
# Size metadata vs Size_igblastn = TODO: where is it ? (should be a unit-level metadata)
# kappa gap 21 = has_kappa_gap_21
# only 20 natural AAs = sequence_aa


import json
import os
import random

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from joblib import Parallel, delayed

from abbert2.oas import OAS
from abbert2.oas import RELATIVE_OAS_TEST_DATA_PATH


def get_one_unit(i_UID, unit, species, chains, UID):
    print(f"\nparsing unit {i_UID} out of {len(UID)}")
    if species != "all" and species.casefold() not in unit.species.casefold():
        print("** discarding unit with species", unit.species)
        df = None
    else:
        df = unit.sequences_df()
        print(f"raw df of size {len(df)}, N heavy={unit.num_heavy_sequences}, N light={unit.num_light_sequences},"
              f" unique chains {df.chain.unique()}")
        if chains != "all":
            df = df[df.chain.str.casefold() == chains.casefold()]
        # TODO: unit.chain can be paired and contain both heavy and light --> fix oas_data_unpaired.py
        if len(df) > 0:
            print(f"kept unit {unit.id} with species {unit.species}")
            df["isotype"] = [unit.isotype] * len(df)
            df["species"] = [unit.species] * len(df)
        else:
            df = None
    return df


def parallel_merge_OAS_df(oas_path, species, chains, save_path, n_jobs, n_break=-1):
    OAS_PATHS = [RELATIVE_OAS_TEST_DATA_PATH,  # = 0 units in the conda env
                 "/project/biomols/antibodies/data/public/oas/20211114",  # the full, unfiltered dataset = 12695 units
                 "/project/biomols/antibodies/data/public/oas/20211114-filters=default"]  # the version we used with cerebras = 12695 units
    OAS_PATH = OAS_PATHS[oas_path]  # we get both heavy and light chains cf unit.id
    oas = OAS(oas_path=OAS_PATH)
    UID = list(oas.units_in_disk())
    random.shuffle(UID)
    if n_break > 0:
        # here n_break is in number of units (not in number of sequences)
        all_df = Parallel(n_jobs=n_jobs)(delayed(get_one_unit)(i_UID, unit, species, chains, UID)
                                    for i_UID, unit in enumerate(UID[:n_break]))
    else:
        all_df = Parallel(n_jobs=n_jobs)(delayed(get_one_unit)(i_UID, unit, species, chains, UID)
                                    for i_UID, unit in enumerate(UID))
    all_df = [df for df in all_df if df is not None]
    print(f"\nfinished parsing {len(all_df)} non-empty valid df")
    merged_df = pd.concat(all_df)
    merged_df.to_parquet(save_path)
    return merged_df


def merge_OAS_df(oas_path, species, chains, save_path, n_break=-1):
    OAS_PATHS = [RELATIVE_OAS_TEST_DATA_PATH,  # = 0 units in the conda env
                 "/project/biomols/antibodies/data/public/oas/20211114",  # the full, unfiltered dataset = 12695 units
                 "/project/biomols/antibodies/data/public/oas/20211114-filters=default"]  # the version we used with cerebras = 12695 units
    OAS_PATH = OAS_PATHS[oas_path]  # we get both heavy and light chains cf unit.id
    oas = OAS(oas_path=OAS_PATH)
    UID = list(oas.units_in_disk())
    random.shuffle(UID)
    merged_df = None
    for i_UID, unit in enumerate(UID):
        print(f"\nparsing unit {i_UID} out of {len(UID)}")
        if species != "all" and species.casefold() not in unit.species.casefold():
            print("** discarding unit with species", unit.species)
        else:
            df = unit.sequences_df()
            print(f"raw df of size {len(df)}, N heavy={unit.num_heavy_sequences}, N light={unit.num_light_sequences},"
                  f" unique chains {df.chain.unique()}")
            if chains != "all":
                df = df[df.chain.str.casefold() == chains.casefold()]
            # TODO: unit.chain can be paired and contain both heavy and light --> fix oas_data_unpaired.py
            if len(df) > 0:
                print(f"kept unit {unit.id} with species {unit.species}")
                df["isotype"] = [unit.isotype]*len(df)
                df["species"] = [unit.species] * len(df)
                if merged_df is None:
                    print(unit.nice_metadata)
                    print(df.columns)
                    df.info()
                    merged_df = df
                else:
                    merged_df = pd.concat([merged_df, df])
                    print(f"current merged_df of size {len(merged_df)}")
            else:
                print("** discarding unit with zero-length or filtered-out chain")

        if n_break > 0 and merged_df is not None and len(merged_df) > n_break:
            break

    merged_df.to_parquet(save_path)
    return merged_df


def seq_replicas_stats(merged_df):
    print("\n\nseq_replicas_stats")
    print("isnull", merged_df.redundancy.isnull().sum())
    print(merged_df.redundancy.describe())
    _seq, _count = np.unique(merged_df.sequence_aa.to_list(), return_counts=True)
    replicas_map = dict(zip(_seq, _count))
    merged_df["n_seq_replicas"] = merged_df["sequence_aa"].map(replicas_map)
    # note: here n_seq_replicas is repeated for each replica
    print(merged_df.n_seq_replicas.describe())
    return merged_df


def cdr3_replicas_stats(merged_df):
    print("\n\ncdr3_replicas_stats")
    merged_df["cdr3_aa"] = [_seq[_cdr3_start: _cdr3_start+_cdr3_length] for _seq, _cdr3_start, _cdr3_length in
                zip(merged_df.sequence_aa.to_list(), merged_df.cdr3_start.to_list(), merged_df.cdr3_length.to_list())]
    _cdr3, _count = np.unique(merged_df.cdr3_aa.to_list(), return_counts=True)
    replicas_map = dict(zip(_cdr3, _count))
    merged_df["n_cdr3_replicas"] = merged_df["cdr3_aa"].map(replicas_map)
    print(merged_df.n_cdr3_replicas.describe())
    return merged_df


def cdr3_len_cutoff_stats(merged_df):
    print("\n\ncdr3_len_cutoff_stats")
    print("isnull", merged_df.anarci_cdr3_is_over_37_aa_long.isnull().sum())
    print(merged_df.anarci_cdr3_is_over_37_aa_long.describe())
    print("all (cdr3_length >= 37) ==  anarci_cdr3_is_over_37_aa_long",
        all((_cdr3_length >= 37) == _is_over_37 for _cdr3_length, _is_over_37 in zip(merged_df.cdr3_length.to_list(),
                                                                merged_df.anarci_cdr3_is_over_37_aa_long.to_list())))
    return merged_df


def region_len_stats(merged_df):
    print("\n\nregion_len_stats")
    for _region in ["cdr1", "cdr2", "cdr3", "fwr1", "fwr2", "fwr3", "fwr4"]:
        print(merged_df[_region+"_length"].describe())
    return merged_df


def indels_stats(merged_df):
    """print(merged_df.has_insertions)  # bool
    print(merged_df.has_unexpected_insertions)  # bool
    print(merged_df.anarci_insertions)  # bool
    print(merged_df.imgt_insertions)  # list
    print(merged_df.anarci_deletions)  # list"""
    print("\n\nindels_stats")
    for col in ["has_insertions", "has_unexpected_insertions", "anarci_insertions", "imgt_insertions", "anarci_deletions"]:
        print(col+".isnull", merged_df[col].isnull().sum())
        if col in ["has_insertions", "has_unexpected_insertions", "anarci_insertions"]:
            print(f"number of {col} {len(merged_df[merged_df[col] == True])} out of {len(merged_df)}")
    n_imgt_insertions = []
    for _imgt_in in merged_df.imgt_insertions.to_list():
        try:
            n_imgt_insertions.append(len("".join(_imgt_in)))
        except:
            n_imgt_insertions.append(0)
    merged_df["n_imgt_insertions"] = n_imgt_insertions
    print(merged_df.n_imgt_insertions.describe())
    assert len(merged_df[merged_df["n_imgt_insertions"] > 0]) == len(merged_df[merged_df["has_insertions"] == True])
    n_anarci_deletions = []
    for _anarci_del in merged_df.anarci_deletions.to_list():
        try:
            n_anarci_deletions.append(len(_anarci_del))
        except:
            n_anarci_deletions.append(0)
    merged_df["n_anarci_deletions"] = n_anarci_deletions
    # merged_df["n_anarci_deletions"] = [len(_anarci_del) for _anarci_del in merged_df.anarci_deletions.to_list()]
    print(merged_df.n_anarci_deletions.describe())
    return merged_df


def truncated_FW_stats(merged_df):
    print("\n\ntruncated_FW_stats")
    for col in ["anarci_fwr1_shorter_than_imgt_defined", "anarci_fwr4_shorter_than_imgt_defined"]:
        print(col+".isnull", merged_df[col].isnull().sum())
        print(merged_df[col].describe())
    for col1, col2, _lt in zip(["fwr1_length", "fwr4_length"], ["FW1_lt_", "FW4_lt_"], [20, 10]):
        merged_df[col2+str(_lt)] = (merged_df[col1] < _lt)
        print(f"{col2+str(_lt)} = {len(merged_df[merged_df[col2+str(_lt)] == True])}")
    return merged_df


def missing_region_stats(merged_df):
    print("\n\nmissing_region_stats")
    for col in ["complete_vdj", "has_wrong_cdr3_reconstruction"]:
        print(col+".isnull", merged_df[col].isnull().sum())
        print(merged_df[col].describe())
    return merged_df


def bulk_stats(merged_df):
    # the isotype column has been added from unit metadata to df column
    print("\n\nbulk_stats")
    print("isnull", merged_df.isotype.isnull().sum())
    merged_df["bulk_isotype"] = (merged_df.isotype.str.casefold() == 'Bulk'.casefold())
    print(f"n bulk_isotype {len(merged_df[merged_df['bulk_isotype'] == True])} out of {len(merged_df)}")
    return merged_df


def productive_stats(merged_df):
    print("\n\nproductive_stats")
    print("isnull", merged_df.productive.isnull().sum())
    print(merged_df.productive.describe())
    return merged_df


def conserved_cysteine_stats(merged_df):
    print("\n\nconserved_cysteine_stats")
    for col in ["anarci_missing_conserved_cysteine", "has_mutated_conserved_cysteines"]:
        print(col+".isnull", merged_df[col].isnull().sum())
        print(merged_df[col].describe())
    return merged_df


def unusual_residue_stats(merged_df):
    print("\n\nunusual_residue_stats")
    print("isnull", merged_df.anarci_unusual_residue.isnull().sum())
    print(merged_df.anarci_unusual_residue.describe())
    return merged_df


def kappa_gap_21_stats(merged_df):
    print("\n\nhas_kappa_gap_21")
    print("isnull", merged_df.has_kappa_gap_21.isnull().sum())
    print(merged_df.has_kappa_gap_21.describe())
    return merged_df


def unnatural_AAs_stats(merged_df, natural_AAs=list("ACDEFGHIKLMNPQRSTVWY")):
    print("\n\nunnatural_AAs_stats")
    unnatural_AAs = []
    for _seq in merged_df.sequence_aa.to_list():
        unnatural_AAs.append(not all(_aa in natural_AAs for _aa in np.unique(list(_seq))))
    merged_df["unnatural_AAs"] = unnatural_AAs
    print(merged_df.unnatural_AAs.describe())
    return merged_df


if __name__ == '__main__':
    # python check_OAS_stats.py --species all --chains all --save_path /home/gnlzm/OAS_datasets/tmp/subsampled25Units_OAS_default_filter --n_break 25 --n_jobs 16
    # python check_OAS_stats.py --species all --chains all --save_path /home/gnlzm/OAS_datasets/tmp/full_OAS_default_filter --n_break -1 --n_jobs 32
    # python check_OAS_stats.py --species human --chains all --save_path /home/gnlzm/OAS_datasets/tmp/human_OAS_default_filter --n_break -1 --n_jobs 32

    parser = ArgumentParser()
    parser.add_argument('--oas_path', default=2, type=int)
    parser.add_argument('--species', default="all", type=str)  # "all", "whatever species"
    parser.add_argument('--chains', default="all", type=str)  # "all", "heavy", "lig
    parser.add_argument('--save_path', default="/home/gnlzm/OAS_datasets/tmp/full_OAS_default_filter", type=str)
    parser.add_argument('--n_break', default=-1, type=int)
    parser.add_argument('--n_jobs', default=1, type=int)
    args = parser.parse_args()

    if not os.path.exists(args.save_path+".parquet"):
        if args.n_jobs > 1:
            print(f"\nrunning parallel_merge_OAS_df with {args.n_jobs} CPUs")
            merged_df = parallel_merge_OAS_df(args.oas_path, args.species, args.chains, args.save_path+".parquet", args.n_jobs, n_break=args.n_break)
        else:
            print("\nrunning merge_OAS_df")
            merged_df = merge_OAS_df(args.oas_path, args.species, args.chains, args.save_path + ".parquet", n_break=args.n_break)
    else:
        print("\nloading pre-computed merged_df")
        merged_df = pd.read_parquet(args.save_path+".parquet")

    merged_df = merged_df.reset_index(drop=True)
    merged_df.info()


    # --> report statistics for various properties of interest

    merged_df = seq_replicas_stats(merged_df)
    merged_df = cdr3_replicas_stats(merged_df)
    merged_df = cdr3_len_cutoff_stats(merged_df)
    merged_df = region_len_stats(merged_df)
    merged_df = indels_stats(merged_df)
    merged_df = truncated_FW_stats(merged_df)
    merged_df = missing_region_stats(merged_df)
    merged_df = bulk_stats(merged_df)
    merged_df = productive_stats(merged_df)
    merged_df = conserved_cysteine_stats(merged_df)
    merged_df = unusual_residue_stats(merged_df)
    merged_df = kappa_gap_21_stats(merged_df)
    merged_df = unnatural_AAs_stats(merged_df, natural_AAs=list("ACDEFGHIKLMNPQRSTVWY"))


    # --> preliminary observations (for a subset of 20211114-filters=default)
    # TODO: double check on full ~300M filtered OAS

    # n_seq_replicas = all 1 (duplicates have been filtered out) but avg redundancy > 1 (study-level)
    # avg n_cdr3_replicas > 1
    # anarci_cdr3_is_over_37_aa_long = all False
    # TODO: is it 37 inclusive or not ?
    # has_insertions and has_unexpected_insertions = not all False
    # n_imgt_insertions and n_anarci_deletions = not all 0
    # FW1_lt_20 and FW4_lt_10 = all False
    # anarci_fwr1_shorter_than_imgt_defined and anarci_fwr4_shorter_than_imgt_defined = some True
    # complete_vdj = some False / has_wrong_cdr3_reconstruction = some True
    # TODO: is complete_vdj proper for light chain (no d_call anyway) ?
    #     print("\n*******************")
    #     print(merged_df[merged_df["complete_vdj"]==True].chain.unique())
    #     print(merged_df[merged_df["complete_vdj"]==False].chain.unique())
    #     print("\n*******************")
    # bulk_isotype = some True
    # productive = all True
    # anarci_missing_conserved_cysteine and has_mutated_conserved_cysteines = all False
    # anarci_unusual_residue = all False
    # has_kappa_gap_21 = all False
    # unnatural_AAs = all False


    # --> plot stats that are not trivial (TODO: add more when working with unfiltered OAS ?)

    merged_df_plotting = merged_df*1

    for _col in ["redundancy", "n_cdr3_replicas", "has_insertions", "has_unexpected_insertions", "n_imgt_insertions",
            "n_anarci_deletions", "anarci_fwr1_shorter_than_imgt_defined", "anarci_fwr4_shorter_than_imgt_defined",
            "complete_vdj", "has_wrong_cdr3_reconstruction", "bulk_isotype",
            "cdr1_length", "cdr2_length", "cdr3_length", "fwr1_length", "fwr2_length", "fwr3_length", "fwr4_length"]:
        print("\nplotting", _col, merged_df[_col].dtypes)
        plt.figure(figsize=(12, 12))
        plt.subplot(211)
        sns.histplot(data=merged_df_plotting, x=_col)
        plt.subplot(212)
        sns.histplot(data=merged_df_plotting, x=_col).set_yscale('log')
        plt.tight_layout()
        plt.savefig(args.save_path+"_"+_col+".jpg")
        plt.close("all")


    # --> apply filtering (TODO: also re-apply the filters from -filters=default)

    print(f"\nunfiltered dataset of size {len(merged_df)}")
    for _filter in [("has_unexpected_insertions", False),
                    ("anarci_fwr1_shorter_than_imgt_defined", False),
                    ("anarci_fwr4_shorter_than_imgt_defined", False),
                    ("has_wrong_cdr3_reconstruction", False)]:
        print("applying filter for", _filter)
        merged_df = merged_df[merged_df[_filter[0]] == _filter[1]]
        print(f"filtered dataset of size {len(merged_df)}")


    # --> store the filtered dataset

    merged_df.to_parquet(args.save_path+"__cleaned.parquet")

