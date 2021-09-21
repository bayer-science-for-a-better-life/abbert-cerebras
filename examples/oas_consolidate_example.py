from abbert2.oas.oas import train_validation_test_iterator

for unit, chain, ml_subset, df in train_validation_test_iterator():
    # Do whatever you want here... save to a consolidated dataset?
    print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)}')
