from abbert2.oas.oas import train_validation_test_iterator, sapiens_like_train_val_test

source_folder="/Users/hossein/Desktop/bayer_sample"
sapiens_like_train_val_test(source_folder)
# for unit, chain, ml_subset, df in train_validation_test_iterator(source_folder):
#     # Do whatever you want here... save to a consolidated dataset?
#     print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}')
