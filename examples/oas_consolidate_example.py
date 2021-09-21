from abbert2.oas.oas import train_validation_test_iterator

dfs= []
counter=0
align_str=["aligned_sequence_heavy","aligned_sequence_light","aligned_sequence_light"]
position_str=["positions_heavy","positions_light","positions_light"]

for unit, chain, ml_subset, df in train_validation_test_iterator():
    # Do whatever you want here... save to a consolidated dataset?
    dframe = unit.sequences_df()
    dframe[align_str[counter]]  = dframe[align_str[counter]].apply(
        lambda x: "-".join(x.astype(str).tolist()))
    dframe[position_str[counter]] = dframe[position_str[counter]].apply(lambda x: '-'.join(x.astype(str).tolist()))
    dframe = dframe.dropna(axis=1)
    dframe.to_csv("../saved_files/{}.csv".format(counter),index=False)

    dfs.append(dframe)
    print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)}')
    counter+=1
print("process finished")

