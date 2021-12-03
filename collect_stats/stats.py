import os
from abbert2.oas.oas import OAS, Unit

from pathlib import Path

import json
import argparse
import sys
import numpy as np
import pandas as pd

def create_arg_parser():
    """
    Create parser for command line args.
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_input_folder",
        required=True,
        help="src_input_folder",
    )
    parser.add_argument(
        "--out_fldr",
        required=True,
        help="out_fldr",
    )

    return parser


def get_sequence(oas_study_path):

    units = []
    for oas_unit_path in Path(oas_study_path).glob('*'):
        if oas_unit_path.is_dir():
            units.extend(list(OAS(str(oas_unit_path)).units_in_path()))

    return units


def get_output_file_name(dest_fldr, oas_subset, study_id):

    out_stats_file_name = f"{oas_subset}_{study_id}_stats.parquet"
    out_stats_file_name = os.path.join(dest_fldr, out_stats_file_name)

    return out_stats_file_name

def get_stats(src_input_folder, out_fldr, seed=1204):
    """
    src_input_folder: ex: /cb/ml/aarti/bayer_sample/paired/Eccles_2020
    """
    print(f"-- src_input_folder: {src_input_folder}")
    print(f"--- out_fldr : {out_fldr}")


    UNITS = get_sequence(oas_study_path=src_input_folder)

    src_input_folder = Path(src_input_folder)
    *_, oas_subset, study_id = src_input_folder.parts

    dest_fldr = out_fldr

    if not os.path.exists(dest_fldr):
        os.makedirs(dest_fldr)

    out_file_name = get_output_file_name(dest_fldr, oas_subset, study_id)

    df_columns = [
        "oas_subset", 
        "study_id", 
        "unit_id",
        "study_year",
        "subject",
        "normalized_species",
        "num_heavy_sequences",
        "num_light_sequences",
        "num_sequences"
        ]
    df_rows = []
    for unit in UNITS:
        metadata = unit.nice_metadata
        row_dict = {}
        for col in df_columns:
            if col not in ["num_heavy_sequences", "num_light_sequences", "num_sequences"]:
                row_dict[col] = getattr(unit, col)
            else:
                unit_df = unit.sequences_df()
                row_dict["num_sequences"] = len(unit_df)
                row_dict["num_heavy_sequences"] = len(unit_df[unit_df["chain"]=="heavy"])
                row_dict["num_light_sequences"] = len(unit_df[unit_df["chain"]=="light"])

        df_rows.append(row_dict)

    stat_df = pd.DataFrame(df_rows)

    stat_df.to_parquet(out_file_name)

def main():
    """
    Main function
    """
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    get_stats(args.src_input_folder, args.out_fldr)

if __name__ == "__main__":
    main()
