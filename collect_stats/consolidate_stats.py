import os
from abbert2.oas.oas import OAS, Unit

from pathlib import Path
import json
import argparse
import sys
import numpy as np
import pandas as pd

def consolidate_stats(src_path, dest_path):
    src_path = Path(src_path)
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
    all_df = pd.DataFrame(columns=df_columns)
    for file in src_path.glob("*.parquet"):
        df = pd.read_parquet(str(file))
        print(f"file: {file}")
        print(df)
        print(all_df)
        print("------")
        all_df = pd.concat([all_df, df], ignore_index=True)
        

    print(all_df)
    all_df.to_parquet(os.path.join(dest_path, "all_stats.parquet"))

    
if __name__ == "__main__":
    src_path = "/cb/customers/bayer/new_datasets/filters_default/stats/stats_parquet"
    dest_path = "/cb/customers/bayer/new_datasets/filters_default/stats/stats_parquet"
    consolidate_stats(src_path, dest_path)

    # src_path = "/cb/home/aarti/ws/code/bayer_tfrecs_filtering/collect_stats/stats"
    # dest_path = "/cb/home/aarti/ws/code/bayer_tfrecs_filtering/collect_stats/stats"
    # consolidate_stats(src_path, dest_path)








        
