from typing import Callable, Optional, Sequence, List

import pandas as pd
from joblib import Parallel, delayed

from abbert2.oas import OAS, Unit
from abbert2.oas.partitioning import assign_ml_subset_by_sequence_hashing

# A subset of columns from the sequences datasets that will be used in our ML training loops
ML_USED_SEQUENCE_COLUMNS = (
    # useful for paired VH/VL (but we need to rethink this in our filtered copies)
    'index_in_unit',
    # loci
    'chain',
    'locus',
    # germlines
    'v_call',
    'd_call',
    'j_call',
    # sequence
    'sequence_aa',
    # regions
    'fwr1_start',
    'fwr1_length',
    'cdr1_start',
    'cdr1_length',
    'fwr2_start',
    'fwr2_length',
    'cdr2_start',
    'cdr2_length',
    'fwr3_start',
    'fwr3_length',
    'cdr3_start',
    'cdr3_length',
    'fwr4_start',
    'fwr4_length',
)

ML_USED_UNIT_COLUMNS = (
    'oas_subset',
    'study_id',
    'unit_id',
    'subject',
    'normalized_species',
)


def assign_ml_subsets(
        unit: Unit,
        partitioner: Callable[[str], Optional[str]] = assign_ml_subset_by_sequence_hashing,
) -> Optional[pd.DataFrame]:
    df = unit.sequences_df(columns=ML_USED_SEQUENCE_COLUMNS)
    if df is None:
        return None  # FIXME: return empty dataframe with the right schema
    df['ml_subset'] = df['sequence_aa'].apply(partitioner)
    for column in ML_USED_UNIT_COLUMNS:
        df[column] = getattr(unit, column)
    return df


def aggregate_stats(
        unit: Unit,
        partitioner: Callable[[str], Optional[str]] = assign_ml_subset_by_sequence_hashing,
        groupers: Sequence[str] = ('normalized_species', 'chain', 'v_call', 'ml_subset',)
) -> Optional[pd.DataFrame]:
    df = assign_ml_subsets(unit=unit, partitioner=partitioner)
    if df is None:
        return None
    return df.groupby(list(groupers) + ['ml_subset']).size().to_frame('num_sequences')


if __name__ == '__main__':

    # collect stats
    units_counts: List[pd.Series] = Parallel(n_jobs=1)(
        delayed(aggregate_stats)(
            unit=unit,
            partitioner=assign_ml_subset_by_sequence_hashing
        )
        for unit in OAS().units_in_disk()
    )

    units_counts = [unit_counts for unit_counts in units_counts if unit_counts is not None]
