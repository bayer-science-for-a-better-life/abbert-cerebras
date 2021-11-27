"""
IMPORTANT NOTE: THIS DOES NOT WORK ANYMORE WITH THE NEWER, SIMPLIFIED UNIT FORMAT
                REWRITE WHEN REEVALUATING

Small benchmark for RAM space, in-disk space and conversion / hashing.

Unfortunately, without further intervention, columns with S1-typed arrays get
parquet-rountripped to object-typed arrays with a bytes object per element.
While compressed in disk this is not a big problem, it makes memory consumption 10x.
Also it makes manipulation (including conversion back to S1) time consuming.

A possibly best solution would be to save these to byte-array types in parquet or binary types in arrow.
  https://arrow.apache.org/docs/python/api/datatypes.html
Include some sort of pandas extension and enable speedy roundtrip.

An even more specialised dtype for these sequences (e.g. using only the needed number of bits)
could be pretty cool.

For the time being, likely, the most convenient approach will be to save strings or bytes.

Notes on fast X
---------------
Tries-like: likely not the best option for our Abs
  https://github.com/pytries/datrie

Parquet vs Feather v2
  https://ursalabs.org/blog/2020-feather-v2/

Hashmap benchmarks
  https://github.com/rurban/smhasher
  https://martin.ankerl.com/2019/04/01/hashmap-benchmarks-01-overview/
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from abbert2.common import to_parquet
from abbert2.oas import OAS
import xxhash

import pyarrow.feather as feather


def load_df(print_info=False):
    unit = OAS().unit('unpaired', 'Khan_2016', 'SRR3175027_Heavy_Bulk')
    df = unit.sequences_df()[['aligned_sequence_heavy']].rename(columns={'aligned_sequence_heavy': 'vh'})
    if print_info:
        print(f'{unit.id}: '
              f'{len(df)} (should be{unit.theoretical_num_sequences_unique}) sequences '
              f'with {sum(df.vh.apply(len))} aas')
    return df


# What about speed, for example, for hashing
def shard(sequences):
    start = time.time()
    shards = [{} for _ in range(256)]
    for sequence in sequences:
        shard = xxhash.xxh32(sequence).intdigest() % 256
        uhash = xxhash.xxh64(sequence).intdigest()
        shards[shard][uhash] = len(shards[shard])
    return time.time() - start


df = load_df(print_info=True)
# 57547 (theory=57547) sequences with 6_595_598 aas

# Our nice S1 dataframe gets converted to a horrible object DF containing a bytes object per amino acid
print(f'(S1-roundtrip) In RAM size {df["vh"].apply(lambda x: x.nbytes).sum()}')  # In RAM size 52_764_784
to_parquet(df, Path.home() / 'from_original.parquet', compression='zstd', compression_level=9)
# In disk size = 787_540
feather.write_feather(df, Path.home() / 'from_original.feather', compression='zstd', compression_level=9)

# How could we enforce the type to be the same?
df['vh'] = df['vh'].apply(lambda x: x.astype('S1', copy=False))  # gets copied anyway, obviously
print(f'(S1) In RAM size {df["vh"].apply(lambda x: x.nbytes).sum()}')  # In RAM size 6_595_598
to_parquet(df, Path.home() / 'S1.parquet', compression='zstd', compression_level=9)
# In disk size = 787540 => AGAIN CONVERSION TOOK PLACE, BUT COMPRESSION MAKES ITS PART
feather.write_feather(df, Path.home() / 'S1.feather', compression='zstd', compression_level=9)

df['vh'] = df['vh'].apply(lambda x: x.tobytes())
print(f'(to-bytes) In RAM size {df.memory_usage(index=False, deep=True)["vh"]}')  # In RAM size 8955025
to_parquet(df, Path.home() / 'to-bytes.parquet', compression='zstd', compression_level=9)
# In disk size = 600673
taken_s = shard(df.vh)
print(f'(to-bytes) {len(df) / taken_s:.2f} sequences / s')
# (to-bytes) 838729.63 sequences / s


# As a small test, what happens if we store these as uint8?
df_uint8 = df.copy()
df_uint8['vh'] = df_uint8['vh'].apply(lambda x: np.array([ord(c) for c in x], dtype='uint8'))
print(f'(uint8) In RAM size {df_uint8["vh"].apply(lambda x: x.nbytes).sum()}')  # In RAM size 6595598
to_parquet(df_uint8, Path.home() / 'uint8.parquet', compression='zstd', compression_level=9)  # In disk size = 787600
df_uint8 = pd.read_parquet(Path.home() / 'uint8.parquet')
print(f'(uint8-roundtrip) In RAM size {df_uint8["vh"].apply(lambda x: x.nbytes).sum()}')  # In RAM size 6595598


# Here as regular python strings
df['vh'] = df['vh'].apply(lambda x: x.tobytes().decode('utf-8'))
print(f'(string) In RAM size {df.memory_usage(index=False, deep=True)["vh"]}')  # In RAM size 9875777
to_parquet(df, Path.home() / 'string.parquet', compression='zstd', compression_level=9)  # In disk size = 600681

# Here as the newer string[pyarrow]
df['vh'] = df['vh'].astype('string[pyarrow]')
print(f'(string-arrow) In RAM size {df.memory_usage(index=False, deep=True)["vh"]}')  # In RAM size 6825790
to_parquet(df, Path.home() / 'string-arrow.parquet', compression='zstd', compression_level=9)  # In disk size = 600681
# Remeber this good old post from Wes: https://arrow.apache.org/blog/2019/02/05/python-string-memory-0.12/
# Lesson should maybe be to live as much in arrow-land as possible

# These get read back as regular python strings
df = pd.read_parquet(Path.home() / 'string-arrow.parquet')
print(f'(string-arrow-roundtrip) In RAM size {df.memory_usage(index=False, deep=True)["vh"]}')  # In RAM size 9875777

# What about converting to bytes?
df['vh'] = df['vh'].apply(lambda x: bytes(x, 'ascii'))
print(f'(bytes) In RAM size {df.memory_usage(index=False, deep=True)["vh"]}')  # In RAM size 8955025

# TODO: Or even better, trully convert to ASCII bytes

taken_s = shard(df.vh)
print(f'(string) {len(df) / taken_s:.2f} sequences / s')
# (string) 708872.33 sequences / s

taken_s = shard(df.vh.astype('string[pyarrow]'))
print(f'(string-arrow) {len(df) / taken_s:.2f} sequences / s')
# (string-arrow) 213325.05 sequences / s
# remember arrow vendors xxhash, so maybe it is possible to actually easily use xxhash against arrow-strings

df = load_df()
taken_s = shard(df.vh)
print(f'(S1-roundtrip) {len(df) / taken_s:.2f} sequences / s')
# (S1-roundtrip) 797028.14 sequences / s

taken_s = shard(df.vh.apply(lambda x: x.astype('S1', copy=False)))
print(f'(S1) {len(df) / taken_s:.2f} sequences / s')
# (S1) 903461.25 sequences / s
# of course, converting takes a lot of time too...

# Of course, some of these representation hash differently... so be careful when swapping
# Could we create a "BIO-SEQUENCE" pandas extension array?
# Could we just live in ARROW land forever?
