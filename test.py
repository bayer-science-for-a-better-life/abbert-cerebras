# from abbert2.oas.oas import train_validation_test_iterator
#
# for unit, chain, ml_subset, df in train_validation_test_iterator():
#     # Do whatever you want here... save to a consolidated dataset?
#     dframe = unit.sequences_df()
#     print(list(df.columns))
#     print(f'unit={unit.id} chain={chain} ml_subset={ml_subset} num_sequences={len(df)} num_columns={len(df.columns)}')

from concurrent.futures import ProcessPoolExecutor
from time import sleep
def task(message):
   sleep(2)
   return message

def main():
   executor = ProcessPoolExecutor(5)
   future = executor.submit(task, ("Completed"))
   print(future.done())
   sleep(2)
   print(future.done())
   print(future.result())
if __name__ == '__main__':
    main()