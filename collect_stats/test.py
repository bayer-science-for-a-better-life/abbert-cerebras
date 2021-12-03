import os
from abbert2.oas.oas import OAS, Unit

from pathlib import Path

import json
import argparse
import sys
import numpy as np
import pandas as pd

txt_path = Path("/cb/customers/bayer/new_datasets/filters_default/stats/bayer_filters_default_for_stats.txt")

with open(txt_path, "r") as fh:
    txt_data = fh.readlines()

txt_data = [os.path.basename(str(x).strip()) for x in txt_data]
print(f"len(files):{len(txt_data)}, {txt_data}")


stats_path = Path("/cb/customers/bayer/new_datasets/filters_default/stats")

files = list(stats_path.glob("*.parquet"))
fname = [str(file.name) for file in files]
fname = [x.split("_stats.parquet")[0] for x in fname]
print(f"fname:{fname}, len: {len(fname)}")


set_fname = set(fname)
set_txtdata = set(txt_data)

print(set_fname, len(set_fname))
print(set_txtdata, len(set_txtdata))
print(set_fname - set_txtdata)
print(set_txtdata - set_fname)

for val in txt_data:
    cnt = txt_data.count(val)
    if cnt > 1:
        print(cnt, val)
