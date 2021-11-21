import time

import pandas as pd
import xxhash

from abbert2.oas import OAS

oas = OAS()

num_shards = 256
shards = {
    'heavy': [{} for _ in range(num_shards)],
    'light': [{} for _ in range(num_shards)],
}

total = 0
index = 0

start = time.time()

for unit in oas.units_in_meta():
    for chain, chain_shards in shards.items():
        df = unit.sequences_df(columns=[f'aligned_sequence_{chain}'])
        if df is None:
            continue
        for sequence in df[f'aligned_sequence_{chain}']:
            total += 1
            sequence = sequence.astype('S1').tobytes().decode('utf-8')
            h = xxhash.xxh3_64_intdigest(sequence)
            shard = chain_shards[h % num_shards]
            if h not in shard:
                shard[h] = index
                index += 1
            if total % 100_000 == 0:
                print(f'{total} ({index} unique) {total / (time.time() - start):.2f} sequences / s')

pd.to_pickle(shards, oas.oas_path / 'uniques.pickle')
