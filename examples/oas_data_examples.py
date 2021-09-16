import json
from pprint import pprint

from abbert2.oas import RELATIVE_OAS_TEST_DATA_PATH, from_parquet
from tf_dataset.dataset_convertor import DatasetConvertor

EXAMPLE_UNIT_PATH = RELATIVE_OAS_TEST_DATA_PATH / 'unpaired' / 'json' / 'Banarjee_2017'
EXAMPLE_UNIT_META_PATH = next(EXAMPLE_UNIT_PATH.glob('*.meta.json'))
EXAMPLE_UNIT_SEQUENCES_PATH = next(EXAMPLE_UNIT_PATH.glob('*.parquet'))

with EXAMPLE_UNIT_META_PATH.open('rt') as reader:
    metadata = json.load(reader)
sequences_df = from_parquet(EXAMPLE_UNIT_SEQUENCES_PATH)

pprint(metadata)
sequences_df.info()
data_set_tf = DatasetConvertor(sequences_df,
                               useless_col=["has_unexpected_insertions", "domain", "insertions", "name",
                                            "original_name",
                                            "has_wrong_sequence_reconstruction", "has_wrong_cdr3_reconstruction"],
                               ordinal_col=["v", "j", "unfit", "has_mutated_conserved_cysteines",
                                            "has_long_cdr1", "has_long_cdr2", "has_long_cdr3", "has_insertions"],
                               list_based_col=["aligned_sequence","positions"]
                               )
data_set_tf.preprocess_dataset()
print("preprocessing phase is finished ....")
data_set_tf.create_tf_dataset()
print("tf dataset created ....")

pprint(sequences_df.iloc[0].to_dict())
