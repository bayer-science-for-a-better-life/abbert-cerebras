"""
Selecting antibodies across the OAS dataset.

Literature
==========

Here it follows a few scerpts from papers relevant to us (one can also check who is citing
the two OAS papers to broaden a bit this view).

There are, of course, other unublished works by Pharma, startups and academia alike
that include other filtering criteria. Of importance is to threshold by redundancy
(e.g., remove all sequences under 3 counts in the same study). We could also extend
this to include redundancy across studies (i.e., add count in our redundancy map).

OAS Update Paper
----------------
From: Observed Antibody Space:
      A diverse database of cleaned, annotated, and translated unpaired and paired antibody sequence
https://onlinelibrary.wiley.com/doi/10.1002/pro.4205

For each sequence, the IMGT numbering scheme was added using antibody numbering and
antigen receptor classIfication (ANARCI) April 23, 2020.
Any sequence that ANARCI could not process was removed.
This step predominantly removes sequences that contain a stop codon.
An ANARCI status highlighting potential problems for each sequence
is retained in the database. This status contains comments regarding unusual residues,
lack of conserved cysteines, deletions and insertions outside of the CDRs,
truncation of frameworks 1 or 4, and if the CDR3 is longer than 37 residues.
Finally, sequences were grouped into units sharing the same metadata,
the same chain (e.g., heavy, light, or paired), and isotype.
---------------


Hu-mAb PAPER
------------
From: Humanization of antibodies using a machine learning approach on large-scale repertoire data
https://academic.oup.com/bioinformatics/article/37/22/4041/6295884
https://www.biorxiv.org/content/10.1101/2021.01.08.425894v2.full

All IgG VH and VL sequences were downloaded from the OAS database (August 2020),
totaling over 500 million sequences in the IMGT format. Human sequences were
split by their V gene type - for example, V1 to V7 for VH sequences.
Redundant sequences, sequences with cysteine errors (18) and sequences
with missing framework 1 residues (residues preceding CDR1) were removed.
The total dataset included over 65 million non-redundant sequences (SI section 1A).
------------


BIOPHI / SAPIENS PAPER
----------------------
From: BioPhi: A platform for antibody design, humanization and humanness evaluation
      based on natural antibody repertoires and deep learning
https://www.biorxiv.org/content/10.1101/2021.08.08.455394v1.full

Unaligned variable region amino acid sequences were downloaded from OAS database (accessed Nov 2019).
A heavy chain training set was extracted by sampling 20 million unaligned redundant amino acid
sequences from all 38 human heavy chain OAS studies from 2011-2017. The training sequences
originated from 24% unsorted, 10% IGHA, 1% IGHD, 1% IGHE, 35% IGHG and 30% IGHM isotypes.
A validation set was extracted by sampling 20 million sequences from all 5 human heavy chain studies
from 2018. The validation sequences originated from 33% unsorted, 16% IGHA, 1% IGHD, 1% IGHE,
20% IGHG and 28% IGHM isotypes. A light chain training set was extracted
by taking all 19,054,615 sequences from all 14 human light chain OAS studies from 2011-2017.
A validation set was extracted by taking all 33,133,386 sequences from both 2 human light
chain OAS studies from 2018. Studies from 2019 were left out to enable future comparison
with new methods on an independent test set.
----------------------


ANTIBERTA PAPER
---------------
From: Deciphering the language of antibodies using self-supervised learning
https://www.biorxiv.org/content/10.1101/2021.11.10.468064v1

Antibody sequences were first filtered out for any sequencing errors, as indicated by OAS.
Sequences were also required to have at least 20 residues before the CDR1, and 10 residues
following the CDR3. The entire collection of 71.98 million unique sequences
(52.89 million unpaired heavy chains and 19.09 million unpaired light chains)
was then split into disjoint training, validation, and test sets using an 80%:10%:10% ratio.
---------------
"""
import time
from typing import Tuple, Sequence, Optional, Union

import numpy as np
import pandas as pd
import xxhash

from abbert2.oas import OAS, Unit


class Filter:

    def __init__(self) -> None:
        super().__init__()
        self.num_units_processed = 0
        self.total_taken_s = 0
        self.num_sequences_processed = 0
        self.num_sequences_filtered_out = 0

    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self, df: pd.DataFrame, unit: Unit = None) -> Tuple[pd.DataFrame, dict]:
        start = time.perf_counter()
        new_df = self._filter(df=df, unit=unit)
        log = {
            'name': self.name,
            'unfiltered_length': len(df),
            'filtered_length': len(new_df),
            'filtered_out': len(df) - len(new_df),
            'taken_s': time.perf_counter() - start,
            'unit': unit.id,
        }
        self.num_units_processed += 1
        self.total_taken_s += log['taken_s']
        self.num_sequences_processed += log['unfiltered_length']
        self.num_sequences_filtered_out += log['filtered_out']
        return new_df, log

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        raise NotImplementedError

    def stat_dict(self) -> dict:
        return {
            'name': self.name,
            'num_units_processed': self.num_units_processed,
            'num_sequences_processed': self.num_sequences_processed,
            'num_sequences_filtered_out': self.num_sequences_filtered_out,
            'num_sequences_processed_per_second': self.num_sequences_processed / self.total_taken_s,
            'taken_s': self.total_taken_s,
        }


class Identity(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df


class MergeRedundant(Filter):

    def __init__(self, use_all_known: bool = False) -> None:
        super().__init__()
        self.use_all_known = use_all_known

    @property
    def name(self):
        return f'MergeRedundant(use_all_known={self.use_all_known})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        if self.use_all_known:
            raise NotImplementedError

        # assume that missing redundancy means 1 (happens in paired)
        df['redundancy'].fillna(value=1, inplace=True)
        estimated_redundancy = df.groupby('sequence_aa')['redundancy'].sum()

        # for the time being, ignore different metadata and keep the first occurrence
        df = df.drop_duplicates('sequence_aa').copy()

        # and assign the estimated redundancy
        df['redundancy'] = estimated_redundancy.loc[df['sequence_aa']].values

        return df

    @staticmethod
    def duplicates_within_same_unit_example():
        # --- VH Example
        # Are there duplicates within the same unit?

        oas = OAS()
        unit = oas.unit('unpaired', 'Gidoni_2019', 'ERR2567201_Heavy_IGHD')
        df = unit.sequences_df()
        num_unique_rows = len(df.drop_duplicates([column for column in df.columns
                                                  if not column.startswith('duplicate_count')
                                                  and not column.startswith('imgt_')
                                                  and not column.startswith('anarci_deletions')
                                                  and not column.startswith('anarci_insertions')]))
        # In general, this should have been done in the nucleotide sequence
        # even if two nucleotide sequences result in the same Ab (synonymous?),
        # they might have different properties (e.g., expressability)
        print(f'Number of examples:    {len(df)}')
        print(f'Number of unique VHs:  {df["sequence_aa_heavy"].nunique()}')
        print(f'Number of unique rows: {num_unique_rows}')
        #
        # Number of examples:    38455  Total Number of Antibodies
        # Number of unique VHs:  38447  There are 8 VHs with duplicates
        # Number of unique rows: 38448  One of the VH with duplicates also differs in another column...
        #
        # Some results are not the same even for the same sequence.
        # Some differences might be coming from nucleotide-based analysis.
        # Unfortunate...
        # Practical solution: aggregate counts and keep a random metadata set
        #


class OnlyProductive(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'productive')


class OnlyIsotypes(Filter):
    # As of 2021/11/26, OAS declares these isotypes at the unit level:
    POSSIBLE_ISOTYPES = 'Bulk', 'All', 'IGHM', 'IGHA', 'IGHE', 'IGHG', 'IGHD'

    # e.g., IgM might be better all filtered out because many never saw an antigen

    def __init__(self, isotypes=None) -> None:
        super().__init__()
        if isotypes is None:
            isotypes = self.POSSIBLE_ISOTYPES
        if isinstance(isotypes, str):
            isotypes = isotypes,
        self.isotypes = set(isotypes)

    @property
    def name(self):
        return f'OnlyIsotypes(isotypes={sorted(self.isotypes)})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        #
        # N.B. this will actually fail, as we would need to put isotype in our dataframes
        #   - isotype is a unit-level field,
        #     as it can only be known if we are targetting a concrete class with our primers
        #   - isotype cannot be inferred from sequence of these VH/VL domains
        #     but the key of this filter is to remove naïve antibodies,
        #     and so the NoNaive filter could be used instead
        #
        # return df[df['isotype'].isin(self.isotypes)]
        #
        # raise NotImplementedError
        if unit.isotype in self.isotypes:
            return df
        else:
            return df.head(0)


# noinspection DuplicatedCode
class NoStopCodon(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not stop_codon')


class VJInFrame(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'vj_in_frame')


class NoUnexpectedInsertions(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not has_unexpected_insertions')


class NoShortFWR1(Filter):

    # "SANGER sequencing is imprecise at the beginning in the begginnig and end"
    # So we want to make sure our sequence is complete and, specially, correct in CDRs

    def __init__(self, threshold: Optional[int] = 20):
        super().__init__()
        self.threshold = threshold

    @property
    def name(self):
        return f'NoShortFWR1(threshold={self.threshold})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        if self.threshold is None:
            return df.query(f'not anarci_fwr1_shorter_than_imgt_defined')
        return df.query(f'fwr1_length >= {self.threshold}')


class NoShortFWR4(Filter):

    # "SANGER sequencing is imprecise at the beginning in the begginnig and end"
    # So we want to make sure our sequence is complete and, specially, correct in CDRs

    def __init__(self, threshold: Optional[int] = 10):
        super().__init__()
        self.threshold = threshold

    @property
    def name(self):
        return f'NoShortFWR4(threshold={self.threshold})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        if self.threshold is None:
            return df.query(f'not anarci_fwr4_shorter_than_imgt_defined')
        return df.query(f'fwr4_length >= {self.threshold}')


class NoKappaGap21(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not has_kappa_gap_21')


class NoDeletions(Filter):
    # Recompute deletions, OAS annotations seem incomplete!
    # Note that some deletions are meant to be present (as IMGT tries to cover all IG domains)
    # See for example:
    #   http://www.imgt.org/IMGTrepertoire/Proteins/proteinDisplays.php?species=human&latin=Homo%20sapiens&group=IGHV
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df[df[f'anarci_deletions'].isna()]


class NoInsertionsOutOfCDRs(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df[df[f'anarci_insertions'].isna()]


# noinspection DuplicatedCode
class NoUnsupportedCDR3Length(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not anarci_cdr3_is_over_37_aa_long')


class NoMissingConservedCysteine(Filter):

    # Check aboss / anarci annotations and original paper to make sure this is correct
    # https://www.jimmunol.org/content/201/12/3694?ijkey=24817c8d879730cb4a170e371cfadd768703b0ed&keytype2=tf_ipsecsha

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not anarci_missing_conserved_cysteine')


class NoUnusualResidues(Filter):
    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'not anarci_unusual_residue')


class CountThreshold(Filter):

    def __init__(self, threshold: int = 2) -> None:
        super().__init__()
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @property
    def name(self):
        return f'CountThreshold(threshold={self.threshold})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        return df.query(f'duplicate_count >= {self.threshold}')


class NoNaive(Filter):
    """
    Count mutations from Germline and remove these with too little
    (naïve B cells that have not been exposed to an antigen and
    therefore not gone through somatic hypermutation).

    To check for somatic hypermutation (to e.g., remove naive B-Cells)
      - Get the germline (e.g., v_call) and map to sequence, as in:
        https://github.com/prihoda/AbNumber/blob/master/abnumber/germlines.py
      - Compare up to the Cys before CDR3 (for both heavy and light)
        This should be done using the numbering
      - Any mutation means the Ab has gone through Somatic HM
        Counting how many may make sense

    A good way to test this is that IGHM repertoires should contain
    an elevated amount of naïve antibodies.
    """

    #
    # B Cell sequenced antibodies
    #

    def __init__(self, threshold: int = 2) -> None:
        super().__init__()
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @property
    def name(self):
        return f'NoNaive(threshold={self.threshold})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        raise NotImplementedError


class NoDuplicates(Filter):
    """
    An online duplicate filter.

    N.B. will build a giagantic inefficient index in memory and needs that all happens in one process...

    To avoid, precompute and redistribute to workers.
    Map in this case to unit_hash, sequence_hash for the chosen sequence.
    """

    def __init__(self) -> None:
        super().__init__()
        self._num_shards = 1024
        self._shards = [set() for _ in range(self._num_shards)]
        self._total = 0
        self._unique = 0

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        no_duplicate = np.ones(len(df), dtype=bool)
        for i, sequence in enumerate(df['sequence_aa']):
            self._total += 1
            if not sequence:  # some rogue paired antibody without this type of chain
                continue
            # noinspection PyArgumentList
            hash0 = xxhash.xxh3_64_intdigest(sequence, seed=0)
            shard = self._shards[hash0 % self._num_shards]
            if hash0 not in shard:
                shard.add(hash0)
                self._unique += 1
            else:
                no_duplicate[i] = False
        return df[no_duplicate]

    def stat_dict(self) -> dict:
        d = super().stat_dict()
        d['num_unique'] = self._unique
        return d


class SimplifySchema(Filter):

    def __init__(self, schema='default') -> None:
        super().__init__()
        self.schema = schema

    @property
    def name(self):
        return f'SimplifySchema(schema={self.schema})'

    def _filter(self, df: pd.DataFrame, unit: Unit = None) -> pd.DataFrame:
        raise NotImplementedError
        # Likely this we can defer to dataset iterators / aggregators


def create_filters_from_name(name: str = 'none') -> Sequence[Filter]:
    return {
        'none': (),

        'default': (
            OnlyProductive(),
            NoShortFWR1(threshold=20),
            NoShortFWR4(threshold=10),
            NoInsertionsOutOfCDRs(),
            NoUnsupportedCDR3Length(),
            NoUnusualResidues(),
            NoMissingConservedCysteine(),
            NoKappaGap21(),
            MergeRedundant(),
            NoDuplicates(),
        ),

        'most-strict': (
            MergeRedundant(),
            CountThreshold(threshold=3),
            OnlyProductive(),
            NoShortFWR1(threshold=20),
            NoShortFWR4(threshold=10),
            NoDeletions(),
            NoInsertionsOutOfCDRs(),
            NoUnsupportedCDR3Length(),
            NoUnusualResidues(),
            NoMissingConservedCysteine(),
            NoKappaGap21(),
            NoDuplicates(),
        ),
    }[name]


def filter_df(df: pd.DataFrame,
              unit: Unit = None,
              *,
              filters: Union[str, Sequence[Filter]] = 'default',
              keep_df_history: bool = False,
              verbose: bool = True):
    if isinstance(filters, str):
        filters = create_filters_from_name(filters)
    logs = []
    filtered_df = df
    for a_filter in filters:
        filtered_df, log = a_filter(df=filtered_df, unit=unit)
        if keep_df_history:
            log['filtered_df'] = filtered_df
        logs.append(log)

    if verbose:
        print_filtering_logs(df, unit, filters, filtered_df, logs)

    return filtered_df, logs


def print_filtering_logs(initial_df, unit, filters, filtered_df, logs):
    print('-' * 80)
    print(f'{unit.id}: from {len(initial_df)} to {len(filtered_df)}')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame(logs)[['name', 'filtered_out', 'taken_s']])
    print('---')
    print('Accummulated Stats')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd.DataFrame([processor.stat_dict() for processor in filters]).round(2))
    print(f'Total units processed: {filters[0].num_units_processed}')
    print(f'Total sequences processed: {filters[0].num_sequences_processed}')
    print(f'Total sequences kept: {filters[-1].num_sequences_processed - filters[-1].num_sequences_filtered_out}')
    print('-' * 80)


if __name__ == '__main__':

    oas = OAS()

    only_test = True

    TEST_UNITS = (
        oas.unit('unpaired', 'Bender_2020', 'ERR3819135_Light_Bulk'),
        oas.unit('unpaired', 'Gidoni_2019', 'ERR2567201_Heavy_IGHD'),
        oas.unit('unpaired', 'Greiff_2017', 'ERR1759628_Heavy_Bulk'),
        oas.unit('paired', 'Alsoiussi_2020', 'SRR11528761_paired'),
        oas.unit('paired', 'Goldstein_2019', 'SRR9179276_paired'),
        oas.unit('paired', 'Goldstein_2019', 'SRR9179276_paired'),
    )

    UNITS = TEST_UNITS if only_test else oas.units_in_disk(oas_subset='unpaired')

    filters = create_filters_from_name('default')

    for unit in UNITS:
        if not unit.has_sequences:
            continue
        df = unit.sequences_df()
        filtered_df, logs = filter_df(df, filters=filters, unit=unit, keep_df_history=False)


#
# TODO: Filters like NoDuplicates and, more prominently, MergeRedundant, break the link between Paired VH/VL
# See: https://github.com/bayer-science-for-a-better-life/abbert-cerebras/issues/7
#
# How to deal with this? =>
# Probably a many to many correspondence between "index_in_csv" of VHs and VLs needs to be kept around.
#
# Not important unless we start training against paired VH & VL sequences
# (e.g., add a self-supervised goal to guess if a pair is real or not alongside
# training with both chains "concatenated")
#
