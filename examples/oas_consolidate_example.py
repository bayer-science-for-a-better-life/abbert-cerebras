from abbert2.oas.oas import train_validation_test_iterator

dfs = []
counter = 0
align_str = ["aligned_sequence_heavy", "aligned_sequence_light"]
position_str = ["positions_heavy", "positions_light"]

for unit, chain, ml_subset, df in train_validation_test_iterator():
    # Do whatever you want here... save to a consolidated dataset?
    dframe = unit.sequences_df()
    print(list(dframe.columns))
    if (counter > 0):
        for element in align_str:
            dframe[element] = dframe[element].apply(
            lambda x: "-".join(x.astype(str).tolist()))
        for element in position_str:
            dframe[element] = dframe[element].apply(lambda x: '-'.join(x.astype(str).tolist()))
    dframe = dframe.dropna(axis=1)
    dframe.to_csv("../saved_files/{}.csv".format(counter), index=False)

    dfs.append(dframe)
    print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)}')
    counter += 1
for i, element in enumerate(dframe.columns):
    print("'"+ element + "' " + ": decoded_row[" + str(i) + "],")

print("********")

# for i, element in enumerate(dframe.columns):
#     print("'" + element + "' : " + element + ",")

print("********")

list_type = []
for element in dframe.columns:
    if (str(dframe[element].dtype) == "int64"):
        list_type.append(int())
    else:
        list_type.append(str())
print(list_type)
