import re
import datetime
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

# --- Proteins

NATURAL_AMINO_ACIDS = 'ABCDEFGHIKLMNPQRSTUVWXYZ'

IUPAC_CODES = (
    ('Ala', 'A'),
    ('Asx', 'B'),
    ('Cys', 'C'),
    ('Asp', 'D'),
    ('Glu', 'E'),
    ('Phe', 'F'),
    ('Gly', 'G'),
    ('His', 'H'),
    ('Ile', 'I'),
    ('Lys', 'K'),
    ('Leu', 'L'),
    ('Met', 'M'),
    ('Asn', 'N'),
    ('Pro', 'P'),
    ('Gln', 'Q'),
    ('Arg', 'R'),
    ('Ser', 'S'),
    ('Thr', 'T'),
    ('Sec', 'U'),
    ('Val', 'V'),
    ('Trp', 'W'),
    ('Xaa', 'X'),
    ('Tyr', 'Y'),
    ('Glx', 'Z')
)


# --- JSON utils

def to_json_friendly(val):

    # Arrays to lists
    if isinstance(val, np.ndarray):
        return list(val)

    # Missings to None
    try:
        if pd.isnull(val):
            return None
    except ValueError:
        ...

    # Identity
    return val


# --- String utils

def to_snake_case(name):
    """Converts strings CamelCased and with spaces into nice python "snake case"."""
    # from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('__([A-Z])', r'_\1', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()


# --- Path / file utils

def mtime(path: Optional[Union[str, Path]]) -> Optional[pd.Timestamp]:
    if path is None:
        return None
    path = Path(path)
    # noinspection PyTypeChecker
    return pd.to_datetime(datetime.datetime.fromtimestamp(path.stat().st_mtime, tz=datetime.timezone.utc))


# --- Parquet conveniences

# noinspection DuplicatedCode
def to_parquet(df: pd.DataFrame,
               path: Union[Path, str],
               compression: Optional[str] = 'zstd',
               compression_level: Optional[int] = 20,
               **write_table_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # noinspection PyArgumentList
    pq.write_table(pa.Table.from_pandas(df), path,
                   compression=compression, compression_level=compression_level,
                   **write_table_kwargs)


# noinspection DuplicatedCode
def from_parquet(path: Union[Path, str], columns=None):
    if columns is not None:
        columns = list(columns)
    return pq.read_table(path, columns=columns).to_pandas()
