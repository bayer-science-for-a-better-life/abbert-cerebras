import os
from pathlib import Path

import numpy as np

# --- Paths

_RELATIVE_DATA_PATH = Path(__file__).parent.parent.parent / 'data'
RELATIVE_OAS_TEST_DATA_PATH = _RELATIVE_DATA_PATH / 'oas-test'
RELATIVE_OAS_FULL_DATA_PATH = _RELATIVE_DATA_PATH / 'oas-full'


def find_oas_path(oas_version='20211114', verbose=False):
    """Try to infer where OAS lives."""

    try:
        from antidoto.data import ANTIDOTO_PUBLIC_DATA_PATH
    except ImportError:
        ANTIDOTO_PUBLIC_DATA_PATH = None

    candidates = (

        # --- Configurable

        # Environment variable first
        os.getenv('OAS_PATH', None),
        # Relative path to oas-full
        RELATIVE_OAS_FULL_DATA_PATH,

        # --- Bayer internal locations

        # Fast local storage in the Bayer computational hosts
        f'/raid/cache/antibodies/data/public/oas/{oas_version}',
        # Default path in the Bayer data lake
        ANTIDOTO_PUBLIC_DATA_PATH / 'oas' / oas_version if ANTIDOTO_PUBLIC_DATA_PATH else None,

        # Test mini-version - better do not activate this and tell out loud we need proper config
        # OAS_TEST_DATA_PATH,
    )

    for candidate_path in candidates:
        if candidate_path is None:
            continue
        candidate_path = Path(candidate_path)
        if candidate_path.is_dir():
            if verbose:
                print(f'OAS data dir: {candidate_path}')
            return candidate_path

    raise FileNotFoundError(f'Could not find the OAS root.'
                            f'\nPlease define the OAS_PATH environment variable '
                            f'or copy / link it to {RELATIVE_OAS_FULL_DATA_PATH}')


# --- Checks

def check_oas_subset(subset: str):
    if subset not in ('paired', 'unpaired'):
        raise ValueError(f'subset should be one of ("paired", "unpaired"), but is {subset}')


# --- Data

def to_chain_independent(df, chains=('heavy', 'light'), add_index=True, add_chain=True):
    if df is None:
        return
    if isinstance(chains, str):
        chains = chains,
    for chain in chains:
        columns = [column for column in df.columns if column.endswith(f'_{chain}')]
        if columns:
            chain_df = df[columns].rename(columns=lambda column: column.rpartition('_')[0])
            if 'chain' in chain_df.columns:
                raise ValueError(f'"chain" should not appear as a column in the dataframe')
            if add_chain:
                chain_df['chain'] = chain
            if add_index:
                chain_df['index'] = np.arange(len(chain_df))
            yield chain, chain_df
