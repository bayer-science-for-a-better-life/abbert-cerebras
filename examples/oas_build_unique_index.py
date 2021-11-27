import time

import xxhash

from abbert2.oas import OAS


def unique_experiment():
    oas = OAS()

    num_shards = 256
    shards = [{} for _ in range(num_shards)]

    total = 0
    index = 0

    start = time.perf_counter()

    for unit in oas.units_in_disk():
        df = unit.sequences_df(columns=[f'sequence_aa'])
        if df is None:
            continue
        for sequence in df[f'sequence_aa']:
            total += 1
            if not sequence:  # some rogue antibody without this type of chain
                continue
            # noinspection PyArgumentList
            hash0 = xxhash.xxh3_64_intdigest(sequence, seed=0)
            # noinspection PyArgumentList
            hash1 = xxhash.xxh3_64_intdigest(sequence, seed=1)
            shard = shards[hash0 % num_shards]
            if hash1 not in shard:
                shard[hash1] = index
                index += 1
            if total % 100_000 == 0:
                print(f'{total} ({index} unique) {int(total / (time.perf_counter() - start))} sequences/s')

    #
    # Exact duplicates, with 99% of the dataset processed:
    #   1_508_100_000 (1_402_842_072 unique) 1_500_000 sequences / s
    #


if __name__ == '__main__':
    unique_experiment()
