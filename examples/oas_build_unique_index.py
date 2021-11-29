import time
from pathlib import Path
from typing import Union

import pandas as pd
import xxhash

from abbert2.oas import OAS


def build_unique_index(oas_path: Union[Path, str] = None):

    oas = OAS(oas_path=oas_path)

    num_shards = 256
    shards = [{} for _ in range(num_shards)]

    total = 0
    unique = 0

    start = time.perf_counter()

    for unit in oas.units_in_disk():
        df = unit.sequences_df(columns=['sequence_aa'])
        if df is None:
            continue
        unit_hash = unit.unit_hash
        for sequence in df['sequence_aa'].unique():
            total += 1
            if not sequence:  # some rogue antibody without this type of chain
                continue
            # noinspection PyArgumentList
            hash0 = xxhash.xxh3_64_intdigest(sequence, seed=0)
            shard = shards[hash0 % num_shards]
            if hash0 not in shard:
                shard[hash0] = unit_hash,  # if we only care about removing dupes, simply store
                unique += 1
            else:
                shard[hash0] += unit_hash,
            if total % 100_000 == 0:
                print(f'{total} ({unique} unique) {int(total / (time.perf_counter() - start))} sequences/s')

    dest_path = oas.oas_path / 'indices'
    dest_path.mkdir(parents=True, exist_ok=True)
    for i, shard in enumerate(shards):
        pd.to_pickle(shard, dest_path / f'duplicates-shard={i}.pickle.xz')

    #
    # Exact duplicates, with 99% of the dataset processed:
    #   1_508_100_000 (1_402_842_072 unique) 1_500_000 sequences / s
    #


if __name__ == '__main__':
    build_unique_index()
