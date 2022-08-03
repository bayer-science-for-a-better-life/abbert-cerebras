import re
import datetime
from pathlib import Path
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq

from anarci.schemes import alphabet as anarci_insertion_alphabet

# --- Proteins

#
# IUPAC Amino Acid Abbreviations
#   https://www.ncbi.nlm.nih.gov/Class/MLACourse/Modules/MolBioReview/iupac_aa_abbreviations.html
# Also look at:
#  https://en.wikipedia.org/wiki/Amino_acid
#  https://en.wikipedia.org/wiki/Amino_acid#/media/File:ProteinogenicAminoAcids.svg
# We add Selenocysteine:
#  https://en.wikipedia.org/wiki/Selenocysteine
#
IUPAC_AMINO_ACIDS = 'ABCDEFGHIKLMNPQRSTUVWXYZ'

NATURAL_AMINO_ACIDS = IUPAC_AMINO_ACIDS.replace('S', '').replace('B', '').replace('Z', '').replace('X', '')

IUPAC_CODES = (
    ('Ala', 'A'),  # Alanine
    ('Asx', 'B'),  # Aspartic acid or Asparagine
    ('Cys', 'C'),  # Cysteine
    ('Asp', 'D'),  # Asparagine
    ('Glu', 'E'),  # Glutamine
    ('Phe', 'F'),  # Phenylalanine
    ('Gly', 'G'),  # Glycine
    ('His', 'H'),  # Histidine
    ('Ile', 'I'),  # Isoleucine
    ('Lys', 'K'),  # Lysine
    ('Leu', 'L'),  # Leucine
    ('Met', 'M'),  # Methonine
    ('Asn', 'N'),  # Asparagine
    ('Pro', 'P'),  # Proline
    ('Gln', 'Q'),  # Glutamine
    ('Arg', 'R'),  # Arginine
    ('Ser', 'S'),  # Serine
    ('Thr', 'T'),  # Threonine
    ('Sec', 'U'),  # Selenocysteine (https://en.wikipedia.org/wiki/Selenocysteine)
    ('Val', 'V'),  # Valine
    ('Trp', 'W'),  # Tryptophan
    ('Xaa', 'X'),  # Any amino acid
    ('Tyr', 'Y'),  # Tyrosine
    ('Glx', 'Z')   # Glutamine or Glutamic acid
)

# --- Antibodies - ANARCI

ANARCI_POSITION_REGEXP = re.compile(r'(\d+)([A-Z]*)')
ANARCI_CODE2INSERTION = [''] + anarci_insertion_alphabet[:-1]  # N.B. swap ' ' with '' (no insertion) and move to start
ANARCI_INSERTION2CODE = {insertion: code for code, insertion in enumerate(ANARCI_CODE2INSERTION)}
ANARCI_INSERTION2CODE[''] = ANARCI_INSERTION2CODE[' '] = 0  # to support also ANARCI ' ' representation of no insertion


def parse_anarci_position(position: str) -> Tuple[int, str]:
    """Parses an anarci position like "112AA" into position (112) and insertion ("AA")."""
    position, insertion = ANARCI_POSITION_REGEXP.match(position).groups()
    return int(position), insertion


def anarci_code_to_insertion(code: int) -> str:
    """Returns the insertion (e.g., "B") corresponding to the code (e.g., 2)."""
    if code is None:
        return ''
    return ANARCI_CODE2INSERTION[code]


def anarci_insertion_to_code(insertion: str) -> int:
    """Returns the code (e.g., 2) corresponding to the insertion (e.g., "B")."""
    return ANARCI_INSERTION2CODE[insertion.strip()]


def parse_anarci_position_to_imgt_code(position_insertion: str) -> Tuple[int, int]:
    """
    Parses an ANARCI position like "112A" into position (112) and insertion code in IMGT (e.g., -1).

    Examples
    --------
    >>> parse_anarci_position_to_imgt_code('1')
    (1, 0)

    >>> parse_anarci_position_to_imgt_code('111')
    (111, 0)

    >>> parse_anarci_position_to_imgt_code('111A')
    (111, 1)

    >>> parse_anarci_position_to_imgt_code('111Z')
    (111, 26)

    >>> parse_anarci_position_to_imgt_code('111AA')
    (111, 27)

    >>> parse_anarci_position_to_imgt_code('111ZZ')
    (111, 52)

    >>> parse_anarci_position_to_imgt_code('112')
    (112, 0)

    >>> parse_anarci_position_to_imgt_code('112A')
    (112, -1)

    >>> parse_anarci_position_to_imgt_code('112Z')
    (112, -26)

    >>> parse_anarci_position_to_imgt_code('112AA')
    (112, -27)

    >>> parse_anarci_position_to_imgt_code('112ZZ')
    (112, -52)
    """
    position, insertion = parse_anarci_position(position_insertion)  # '112AA' -> 112, 'AA'
    insertion = anarci_insertion_to_code(insertion)                  # 'AA' ->

    # in IMGT insertions are sorted in reverse for position 112
    if position == 112:
        insertion = -insertion

    return position, insertion


def anarci_imgt_code_to_insertion(code: int) -> str:
    """Returns the insertion (e.g., "B") corresponding to the code (e.g., -2)."""
    if code is None:
        return ''
    return ANARCI_CODE2INSERTION[abs(code)]


def parse_anarci_position_aa_to_imgt_code(position_aa) -> Tuple[int, int, str]:
    """
    Parses a position=aa string (e.g., 112A=V) to a tuple (112, -1, "V").

    Examples
    --------
    >>> parse_anarci_position_aa_to_imgt_code('1=A')
    (1, 0, 'A')

    >>> parse_anarci_position_aa_to_imgt_code('111ZZ=V')
    (111, 52, 'V')

    >>> parse_anarci_position_aa_to_imgt_code('112A=V')
    (112, -1, 'V')
    """
    position, aa = position_aa.split('=')
    position, insertion = parse_anarci_position_to_imgt_code(position)
    return position, insertion, aa


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
    name = re.sub('([a-z\d])([A-Z])', r'\1_\2', name)
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
               compression_level: Optional[int] = 9,
               preserve_index=True,
               **write_table_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # noinspection PyArgumentList
    pq.write_table(pa.Table.from_pandas(df, preserve_index=preserve_index), path,
                   compression=compression, compression_level=compression_level,
                   **write_table_kwargs)


# noinspection DuplicatedCode
def from_parquet(path: Union[Path, str], columns=None, as_dataframe=True):
    if isinstance(columns, str):
        columns = columns,
    if columns is not None:
        columns = list(columns)
    table = pq.read_table(path, columns=columns)
    if as_dataframe:
        return table.to_pandas()
    return table


# --- Combinatorials

def number_of_mutants(
    sequence_length: int = 3,
    max_mutations: int = None,
    alphabet_size: int = len(NATURAL_AMINO_ACIDS)
) -> int:
    """
    Returns the number of mutants for a sequence of the provided length when
    only a maximum number of positions are allowed to mutate at the same time.

    Parameters
    ----------
    sequence_length : int, default 3
      The length of the sequence

    max_mutations : int or None, default None
      The maximum number of positions to mutate.
      If None, `sequence_length` is used and this reduces to alphabet_size^sequence_length
      Otherwise, it must be in [1, sequence_length]

    alphabet_size : int, default 20
      The number of different values each position can adopt

    Examples
    --------
    # When no mutation is allowed, we are left with the original sequence
    >>> number_of_mutants(sequence_length=3, max_mutations=0)
    1

    # When only one position at a time is allowed to mutate, we have 20 * 3 = 60
    >>> number_of_mutants(sequence_length=3, max_mutations=1)
    60

    # When maximum two positions at a time are allowed to mutate, we have 2^20 * 3! / (2! (3 - 2)!) = 1_200
    >>> number_of_mutants(sequence_length=3, max_mutations=2)
    1200

    # When all positions are allowed to mutate at a time, we have 20^3 = 8_000
    >>> number_of_mutants(sequence_length=3, max_mutations=None)
    8000
    """
    import math
    if max_mutations is None:
        max_mutations = sequence_length
    if sequence_length < max_mutations:
        raise ValueError(f'Number of positions to mutate must')
    return math.comb(sequence_length, max_mutations) * alphabet_size ** max_mutations
