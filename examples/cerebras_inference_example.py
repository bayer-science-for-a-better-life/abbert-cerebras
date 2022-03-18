import time

import pandas as pd

from abbert2.abbert2 import InfiniteAbbert2

if __name__ == '__main__':

    run_id = 15
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    start = time.perf_counter()
    embedder = InfiniteAbbert2(run_id=run_id, warmup=True)
    print(f'Setting-up: {time.perf_counter() - start:.2f}s')
    # This will take long everywhere

    start = time.perf_counter()
    embeddings_df = pd.DataFrame(embedder(sequences))
    print(f'Inference: {time.perf_counter() - start:.2f}s')
    # This will take very long on CPU, very little on GPU

    embeddings_df.info()
    #
    # Data columns (total 15 columns):
    #  #   Column                   Non-Null Count  Dtype
    # ---  ------                   --------------  -----
    #  0   sequence                 3 non-null      object
    #  1   features/input_ids       3 non-null      object
    #  2   features/input_mask      3 non-null      object
    #  3   encoder_layer_0_output   3 non-null      object
    #  4   encoder_layer_1_output   3 non-null      object
    #  5   encoder_layer_2_output   3 non-null      object
    #  6   encoder_layer_3_output   3 non-null      object
    #  7   encoder_layer_4_output   3 non-null      object
    #  8   encoder_layer_5_output   3 non-null      object
    #  9   encoder_layer_6_output   3 non-null      object
    #  10  encoder_layer_7_output   3 non-null      object
    #  11  encoder_layer_8_output   3 non-null      object
    #  12  encoder_layer_9_output   3 non-null      object
    #  13  encoder_layer_10_output  3 non-null      object
    #  14  encoder_layer_11_output  3 non-null      object
    #
