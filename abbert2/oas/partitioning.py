"""Partitioning strategies for machine learning evaluation."""
import pandas as pd
import xxhash


# These would be interned anyway, but I like explicit
SUBSET_TRAIN = 'train'
SUBSET_VALIDATION = 'validation'
SUBSET_TEST = 'test'


#
# --- Sequence based hashing
#
# Pros:
#   - Simple
#   - Fast
#   - Reproducible "on the fly"
#   - Will keep priors
# Cons:
#   - Sensitive to possible batch effects
#     (sequencing run AKA unit, subject, study)
#   - No cross-val
#


def assign_ml_subset_by_sequence_hashing(sequence: str,
                                         seed: int = 0,
                                         train_pct: int = 80,
                                         val_pct: int = 10,
                                         test_pct: int = 10):

    if (train_pct + val_pct + test_pct) != 100:
        raise ValueError(f'train_pct + val_pct + test_pct must equal 100, '
                         f'but it is {(train_pct + val_pct + test_pct)}')

    if pd.isnull(sequence):
        return None

    # noinspection PyArgumentList
    bucket = xxhash.xxh32_intdigest(sequence, seed=seed) % 100
    if bucket < train_pct:
        return SUBSET_TRAIN
    if bucket < (train_pct + val_pct):
        return SUBSET_VALIDATION
    return SUBSET_TEST
