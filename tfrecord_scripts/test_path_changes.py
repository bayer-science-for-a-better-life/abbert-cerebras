from tfrecord_scripts.create_tf_records import get_sequence

from abbert2.oas.oas_original import OAS
from abbert2.common import to_json_friendly, from_parquet
import os

file_path = "/cb/ml/aarti/bayer_sample_new_datasets"

os.environ["OAS_PATH"] = file_path

UNITS = list(OAS().units_in_disk())

# for unit in UNITS:
#     print(dir(unit))
#     print(unit.nice_metadata)
#     print(unit.chain)


samples = sorted(UNITS, key=lambda x: x.sequences_path.parent)

folder_paths = list(set([x.sequences_path.parent for x in samples]))
samples_new = []
for fldr in folder_paths:
    gen_df = get_sequence(oas_path=str(fldr))
    samples_new.extend(list(gen_df))

samples_new = sorted(samples_new, key=lambda x: x.sequences_path.parent)

print(len(samples))
print(len(samples_new))

for i, unit in enumerate(samples_new):
    df_new = unit.sequences_df()
    df = samples[i].sequences_df()
    print(df.info())
    # assert df_new.equals(df)
    break


# for i, item in enumerate(samples):
#     unit, chain, ml_subset, df = item
#     print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}, {unit.sequences_path}')

#     unit_new, chain_new, ml_subset_new, df_new = samples_new[i]
#     print(f"unit={unit_new.id} chain={chain_new} ml_subset={ml_subset_new} num_sequences={len(df_new)} num_columns={len(df_new.columns)}, {unit_new.sequences_path}")

#     assert unit.id == unit_new.id
#     assert chain == chain_new
#     assert ml_subset == ml_subset_new
#     assert df.sort_index().equals(df_new.sort_index())

#     print("----0000000-------")




