import json
from pprint import pprint
import tensorflow as tf
from abbert2.oas import RELATIVE_OAS_TEST_DATA_PATH, from_parquet

EXAMPLE_UNIT_PATH = RELATIVE_OAS_TEST_DATA_PATH / 'unpaired' / 'json' / 'Banarjee_2017'
EXAMPLE_UNIT_META_PATH = next(EXAMPLE_UNIT_PATH.glob('*.meta.json'))
EXAMPLE_UNIT_SEQUENCES_PATH = next(EXAMPLE_UNIT_PATH.glob('*.parquet'))

with EXAMPLE_UNIT_META_PATH.open('rt') as reader:
    metadata = json.load(reader)
sequences_df = from_parquet(EXAMPLE_UNIT_SEQUENCES_PATH)

print(sequences_df.head())

sequences_df["aligned_sequence"] = sequences_df["aligned_sequence"].apply(lambda x: "-".join(x.astype(str).tolist()))

sequences_df["positions"] = sequences_df["positions"].apply(lambda x: '-'.join(x.astype(str).tolist()))

sequences_df = sequences_df.dropna(axis=1)

sequences_df.to_csv("/cb/home/aarti/ws/code/bayer_poc_monolith/src/customer_models/bayer/bert/tf/data/pq_converted.csv", index=False)
