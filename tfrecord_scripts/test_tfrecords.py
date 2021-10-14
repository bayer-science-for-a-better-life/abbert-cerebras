from tfrecord_scripts.create_tf_records import get_iterator

from abbert2.oas.oas_actual import train_validation_test_iterator

file_path = "/cb/ml/aarti/bayer_sample"


samples = list(train_validation_test_iterator())

samples = sorted(samples, key=lambda x: x[0].sequences_path.parent)

folder_paths = list(set([x[0].sequences_path.parent for x in samples]))
samples_new = []
for fldr in folder_paths:
    iterator_tf = get_iterator(oas_path=str(fldr))
    samples_new.extend(list(iterator_tf))


samples_new = sorted(samples_new, key=lambda x: x[0].sequences_path.parent)

print(len(samples))
print(len(samples_new))

for i, item in enumerate(samples):
    unit, chain, ml_subset, df = item
    print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}, {unit.sequences_path}')

    unit_new, chain_new, ml_subset_new, df_new = samples_new[i]
    print(f"unit={unit_new.id} chain={chain_new} ml_subset={ml_subset_new} num_sequences={len(df_new)} num_columns={len(df_new.columns)}, {unit_new.sequences_path}")

    assert unit.id == unit_new.id
    assert chain == chain_new
    assert ml_subset == ml_subset_new
    assert df.sort_index().equals(df_new.sort_index())

    print("----0000000-------")






        






