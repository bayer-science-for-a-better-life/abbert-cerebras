from abbert2.oas.oas import train_validation_test_iterator
from abbert2.oas.oas import sapiens_like_train_val_test
import numpy as np



# for unit, chain, ml_subset, df in train_validation_test_iterator():
#     # Do whatever you want here... save to a consolidated dataset?
#     print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}')


oas_path = "/cb/ml/aarti/bayer_sample/paired/Eccles_2020/SRR10358525_paired"
# oas_path = "/cb/ml/aarti/bayer_sample/paired/Alsoiussi_2020/SRR11528762_paired"

def partioner(oas_path):
    return lambda : sapiens_like_train_val_test(oas_path=oas_path)

iterator = train_validation_test_iterator(partitioner=partioner(oas_path))

for unit, chain, ml_subset, df in iterator:
    # Do whatever you want here... save to a consolidated dataset?
    if all([chain, ml_subset, not df.empty]):
        print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}, {unit.sequences_path}')

        print(df.columns)
        print(dir(unit))
        print(df[f'cdr1_start_{chain}'].astype(np.int64))
        # for col in df.columns:
        #     print(col)
        
        # query = ""
        # for col in df.columns:
        #     if "aligned_sequence" not in col and "fw" not in col:
        #         query = query + f"{col} >= 25 and "

        # print(query[:-5], query[:-5][-1])
        
        # df = df.query(query[:-5])
        # print(df)
    else:
        print(f'unit={unit.id}')