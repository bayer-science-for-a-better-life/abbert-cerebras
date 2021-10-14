
from abbert2.oas.oas_actual import train_validation_test_iterator
from tfrecord_scripts.create_tf_records import get_iterator

for unit, chain, ml_subset, df in train_validation_test_iterator():
    # Do whatever you want here... save to a consolidated dataset?
    print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}')

    iterator_tf = get_iterator(oas_path=str(unit.sequences_path.parent))

    for unit_new, chain_new, ml_subset_new, df_new in iterator_tf:
        print(f"unit={unit_new.id} chain={chain_new} ml_subset={ml_subset_new} num_sequences={len(df_new)} num_columns={len(df_new.columns)}, {unit_new.sequences_path}")

    print("------------")