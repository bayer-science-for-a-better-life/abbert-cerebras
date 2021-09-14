import json
from pprint import pprint

from abbert2.oas import RELATIVE_OAS_TEST_DATA_PATH, from_parquet

EXAMPLE_UNIT_PATH = RELATIVE_OAS_TEST_DATA_PATH / 'unpaired' / 'json' / 'Banarjee_2017'
EXAMPLE_UNIT_META_PATH = next(EXAMPLE_UNIT_PATH.glob('*.meta.json'))
EXAMPLE_UNIT_SEQUENCES_PATH = next(EXAMPLE_UNIT_PATH.glob('*.parquet'))

with EXAMPLE_UNIT_META_PATH.open('rt') as reader:
    metadata = json.load(reader)
sequences_df = from_parquet(EXAMPLE_UNIT_SEQUENCES_PATH)

pprint(metadata)
sequences_df.info()
pprint(sequences_df.iloc[0].to_dict())
