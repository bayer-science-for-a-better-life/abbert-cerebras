import time

import pandas as pd
import xxhash

from abbert2.oas import OAS


def unique_experiment():
    oas = OAS()

    num_shards = 256
    shards = [{} for _ in range(num_shards)]

    total = 0
    index = 0

    start = time.time()

    for unit in oas.units_in_disk():
        for chain in ('heavy', 'light'):
            df = unit.sequences_df(columns=[f'sequence_aa_{chain}'])
            if df is None:
                continue
            for sequence in df[f'sequence_aa_{chain}']:
                total += 1
                if not sequence:  # some rogue antibody without this type of chain
                    continue
                sequence = sequence.astype('S1').tobytes().decode('utf-8')
                h = xxhash.xxh3_64_intdigest(sequence)
                shard = shards[h % num_shards]
                if h not in shard:
                    shard[h] = index
                    index += 1
                if total % 100_000 == 0:
                    print(f'{total} ({index} unique) {total / (time.time() - start):.2f} sequences / s')
    pd.to_pickle(shards, oas.oas_path / 'uniques.pickle')

    #
    # Exact duplicates, with 99% of the dataset processed:
    #   1_508_100_000 (1_402_842_072 unique) 81000.00 sequences / s
    #
