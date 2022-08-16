# script to load OAS and retrieve statistics for various properties / issues

# N replicas (e.g. >3), CDR3 replicas, too long region lengths (e.g. CDR3 length cutoff of 37)
# anarci status, unusual insertions/deletions, truncated beginning (FW1>=20) or ending (FW4>=10)
# missing regions, bulk assignment (e.g. too low isotype assignment, short or missing CH1)
# productive sequence, lack of conserved cysteines, unusual residues, Size metadata vs Size_igblastn
# kappa gap 21 = True, only 20 natural AAs
# missing values

# TODO: later, when available, compute stats for maturity related properties e.g. identity to germline


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

from abbert2.oas import OAS
from abbert2.oas import RELATIVE_OAS_TEST_DATA_PATH


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
                if merged_df is None:
                    print(unit.nice_metadata)
                    df.info()
                    merged_df = df
                else:
                    merged_df = pd.concat([merged_df, df])
                    print(f"current merged_df of size {len(merged_df)}")
            else:
                print("** discarding unit with zero-length or filtered-out chain")

        if n_break > 0 and len(merged_df) > n_break:
            break

    merged_df.to_parquet(save_path)
    return merged_df


if __name__ == '__main__':
    # python check_OAS_stats.py --species all --chains all
    # python check_OAS_stats.py --species all --chains all --save_path /home/gnlzm/OAS_datasets/tmp/subsampled100k_OAS_default_filter.parquet --n_break 100000

    parser = ArgumentParser()
    parser.add_argument('--oas_path', default=2, type=int)
    parser.add_argument('--species', default="all", type=str)  # "all", "whatever species"
    parser.add_argument('--chains', default="all", type=str)  # "all", "heavy", "lig
    parser.add_argument('--save_path', default="/home/gnlzm/OAS_datasets/tmp/full_OAS_default_filter.parquet", type=str)
    parser.add_argument('--n_break', default=-1, type=int)
    args = parser.parse_args()

    natural_AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    if not os.path.exists(args.save_path):
        print("\nrunning merge_OAS_df")
        merged_df = merge_OAS_df(args.oas_path, args.species, args.chains, args.save_path, n_break=args.n_break)
    else:
        print("\nloading pre-computed merged_df")
        merged_df = pd.read_parquet(args.save_path)

    merged_df.info()

    # TODO: add columns based on criteria
    # TODO: make stats and visualizations from stored df

