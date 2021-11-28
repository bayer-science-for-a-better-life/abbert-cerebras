import time
from pathlib import Path

import pandas as pd

from abbert2.common import to_parquet, from_parquet
from abbert2.oas import OAS


def bench_sequences_compression():

    # poor benchmark to get a small idea of what works best...

    oas = OAS()

    bench_results = []

    for unit in oas.units_in_disk(oas_subset='unpaired'):

        if not unit.has_sequences:
            continue
        df = unit.sequences_df()

        # original: 3_090_253
        dest_path = Path.home() / '00-original.parquet'
        to_parquet(df, dest_path, preserve_index=False)
        bench_results.append({
            'unit': '-'.join(unit.id),
            'name': 'original',
            'compressor': 'zstd',
            'level': 9,
            'compress_time_s': None,
            'decompress_time_s': None,
            'size_MiB': dest_path.stat().st_size / 1024 ** 2
        })
        dest_path.unlink()

        # sorting by sequence helps a bit: 2_714_116
        df = df.sort_values(['sequence_aa'])
        dest_path = Path.home() / '01-df.sorted.parquet'
        to_parquet(df, dest_path, preserve_index=False)
        bench_results.append({
            'unit': '-'.join(unit.id),
            'name': 'sorted',
            'compressor': 'zstd',
            'level': 9,
            'compress_time_s': None,
            'decompress_time_s': None,
            'size_MiB': dest_path.stat().st_size / 1024 ** 2
        })
        dest_path.unlink()

        # removing hashes helps a lot: 1_562_750
        df = df.drop(columns=['hash0', 'hash1'])
        dest_path = Path.home() / '02-df.sorted.nohashes.parquet'
        to_parquet(df, dest_path, preserve_index=False)
        bench_results.append({
            'unit': '-'.join(unit.id),
            'name': 'sorted-nohashes',
            'compressor': 'zstd',
            'level': 9,
            'compress_time_s': None,
            'decompress_time_s': None,
            'size_MiB': dest_path.stat().st_size / 1024 ** 2
        })
        dest_path.unlink()

        gzips = (
            ('gzip', 1),
            ('gzip', 2),
            ('gzip', 3),
            ('gzip', 4),
            ('gzip', 5),
            ('gzip', 6),
            ('gzip', 7),
            ('gzip', 8),
            ('gzip', 9),
        )

        brotlis = (
            ('brotli', 1),
            ('brotli', 2),
            ('brotli', 3),
            ('brotli', 4),
            ('brotli', 5),
            ('brotli', 6),
            ('brotli', 7),
            ('brotli', 8),
            ('brotli', 9),
            ('brotli', 10),
            ('brotli', 11),
            ('brotli', 12),
        )

        # At the moment level cannot be used with parquet
        # See: https://issues.apache.org/jira/browse/ARROW-9648
        lz4s = (
            ('lz4', None),
            # ('lz4', 1),
            # ('lz4', 2),
            # ('lz4', 3),
            # ('lz4', 4),
            # ('lz4', 5),
            # ('lz4', 6),
            # ('lz4', 7),
            # ('lz4', 8),
            # ('lz4', 9),
            # ('lz4', 10),
            # ('lz4', 11),
            # ('lz4', 12),
            # ('lz4', 13),
            # ('lz4', 14),
            # ('lz4', 15),
            # ('lz4', 16),
        )

        zstds = (
            ('zstd', -10),
            ('zstd', -9),
            ('zstd', -8),
            ('zstd', -7),
            ('zstd', -6),
            ('zstd', -5),
            ('zstd', -4),
            ('zstd', -3),
            ('zstd', -2),
            ('zstd', -1),
            ('zstd', 1),
            ('zstd', 2),
            ('zstd', 3),
            ('zstd', 4),
            ('zstd', 5),
            ('zstd', 6),
            ('zstd', 7),
            ('zstd', 8),
            ('zstd', 9),
            ('zstd', 10),
            ('zstd', 11),
            ('zstd', 12),
            ('zstd', 13),
            ('zstd', 14),
            ('zstd', 15),
            ('zstd', 16),
            ('zstd', 17),
            ('zstd', 18),
            ('zstd', 19),
            ('zstd', 20),
            ('zstd', 21),
            ('zstd', 21),
            ('zstd', 22),
            ('zstd', 23),
        )

        compressors = zstds + gzips + brotlis + lz4s

        # what about compression levels?
        for compressor, level in compressors:
            dest_path = Path.home() / f'03-compressor={compressor}-level={level}.parquet'
            start = time.perf_counter()
            to_parquet(df, dest_path,
                       compression=compressor,
                       compression_level=level,
                       preserve_index=False)
            compress_time = time.perf_counter() - start
            start = time.perf_counter()
            from_parquet(dest_path)
            decompress_time = time.perf_counter() - start
            bench_results.append({
                'unit': '-'.join(unit.id),
                'name': f'sorted-nohashes-{compressor}={level}',
                'compressor': compressor,
                'level': level,
                'compress_time_s': compress_time,
                'decompress_time_s': decompress_time,
                'size_MiB': dest_path.stat().st_size / 1024 ** 2
            })
            dest_path.unlink()
            print(bench_results[-1])
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     print(pd.DataFrame(bench_results).groupby('unit').idmax('size_MiB'))

    df = pd.DataFrame(bench_results)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df.round(2))
    df.to_pickle(str(Path.home() / 'benchmark.pickle'))


if __name__ == '__main__':
    bench_sequences_compression()
