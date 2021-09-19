"""
Observed Antibody Space data manipulation.

OAS home:
  http://opig.stats.ox.ac.uk/webapps/oas/
  (equivalently see antibodymap.org)

OAS paper:
  https://www.jimmunol.org/content/201/8/2502

OAS OPIG Blogposts:
  - https://www.blopig.com/blog/2018/06/how-to-parse-oas-data/
  - https://www.blopig.com/blog/2018/11/new-data-in-oas/
  - https://www.blopig.com/blog/2020/07/observed-antibody-space-miairr/
  - https://www.blopig.com/blog/2020/09/adding-paired-bcr-data-to-oas/

OAS is the baby of Aleksandr Kovaltsuk, whom I find useful to follow:
  - Twitter: https://twitter.com/antibodymap?lang=en
  - Home: https://konradkrawczyk.github.io/
  - Company: https://naturalantibody.com/
  - Other OPIG Blogposts: https://www.blopig.com/blog/author/aleksandr/

Some code and papers looking at OAS (scholar reports ~50 citations to the OAS paper 2021/08):
  - BioPhi
  - https://doi.org/10.1101/2021.01.08.425894
  - https://github.com/dahjan/OAS-MiXCR-pipeline
  - https://github.com/dahjan/OAS-data-visualization

AIRR Standards:
  https://docs.airr-community.org/en/stable/miairr/introduction_miairr.html

The origin of it all:
  https://www.ncbi.nlm.nih.gov/sra
"""
import gzip
import json
import os
import time
from ast import literal_eval
from functools import cached_property
from itertools import chain, product
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from more_itertools import ichunked
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
from pyarrow import parquet as pq


# --- Paths

_RELATIVE_DATA_PATH = Path(__file__).parent.parent / 'data'
RELATIVE_OAS_TEST_DATA_PATH = _RELATIVE_DATA_PATH / 'oas'
RELATIVE_OAS_FULL_DATA_PATH = _RELATIVE_DATA_PATH / 'oas-full'


def find_oas_path(verbose=False):
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
        '/raid/cache/antibodies/data/public/oas/20210717',
        # Default path in the Bayer data lake
        ANTIDOTO_PUBLIC_DATA_PATH / 'oas' / '20210717' if ANTIDOTO_PUBLIC_DATA_PATH else None,

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


# --- Utils

# noinspection PyDefaultArgument
def _initialize_dask(dask_client_address=None, client=[]):

    from dask.distributed import Client, LocalCluster
    import dask

    if dask_client_address is None:
        dask_client_address = 'tcp://localhost:8786'

    dask.config.set({'distributed.worker.daemon': False})

    if not client:
        try:
            # noinspection PyTypeChecker
            client.append(Client(dask_client_address, timeout='2s'))
        except OSError:
            cluster = LocalCluster(
                processes=True,
                scheduler_port=int(dask_client_address.rpartition(':')[2])
            )
            client.append(Client(cluster))


# --- OAS Management

def parse_oas_csv_unit(oas_unit_path: Union[str, Path],
                       only_meta=False) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Parses an OAS unit in the new CSV format.

    See: https://www.blopig.com/blog/2020/07/observed-antibody-space-miairr/
    """
    with gzip.open(oas_unit_path) as reader:
        metadata = json.loads(next(reader))
    if not only_meta:
        return metadata, pd.read_csv(oas_unit_path, sep=',', skiprows=1, low_memory=False)
    return metadata, None


def _parse_antibody_jsons(json_iterator: Iterator[str],
                          keep_errors: bool,
                          origin: str,
                          continue_on_error: bool = False):
    antibodies = []
    numbering_columns = None

    for antibody in json_iterator:
        try:
            # parse json
            antibody = json.loads(antibody)
            # take care of OAS reported errors, that are heavy to keep around
            if keep_errors:
                antibody['errors'] = [list(error) if error is not None else None
                                      for error in literal_eval(antibody['errors'])]
            else:
                del antibody['errors']
            try:
                antibody['num_errors'] = literal_eval(antibody['num_errors'])
            except ValueError:
                antibody['num_errors'] = None
            # preprocess, check and summarize numbering
            preprocessed_numbering = preprocess_anarci_data(
                numbering_data_dict=json.loads(antibody['data']),
                expected_sequence=antibody['seq'],
                expected_cdr3=antibody['cdr3']
            )
            if numbering_columns is None:
                numbering_columns = list(preprocessed_numbering)
            antibody.update(preprocessed_numbering)
            # get rid of redundant data
            del antibody['data']
            del antibody['seq']
            del antibody['cdr3']
            # add to the datast
            antibodies.append(antibody)
        except Exception as ex:
            print(f'ERROR for antibody {antibody}: {str(ex)}')
            if not continue_on_error:
                raise

    df = pd.DataFrame(antibodies)

    # massage dataframe and report unexpected columns

    expected_columns = [
        'name',
        'original_name',
        'v',
        'j',
        'redundancy',
        'num_errors',
        'errors',
    ] + numbering_columns

    optional_columns = ['errors']
    present_columns = list(df.columns)

    missing_columns = sorted(set(expected_columns) - set(present_columns + optional_columns))
    if missing_columns:
        print(f'\tCOLUMNS MISSING ({missing_columns}) FROM {origin}')

    unexpected_columns = sorted(set(present_columns) - set(expected_columns))
    if unexpected_columns:
        print(f'\tCOLUMNS UNEXPECTED ({unexpected_columns}) FROM {origin}')

    columns = [column for column in expected_columns if column in df.columns]
    columns += [column for column in df.columns if column not in expected_columns]

    return df[columns]


def parse_oas_json_unit(oas_unit_path: Union[str, Path],
                        only_meta=False,
                        keep_errors=False,
                        n_jobs=8,
                        batch_size=10_000,
                        continue_on_error=True) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Parses an OAS unit in the old JSON format.

    See: https://www.blopig.com/blog/2018/06/how-to-parse-oas-data/
    """
    # TODO: if this is too slow, use a faster json parser allowing partial unmarshalling, like pysimdjson

    with gzip.open(oas_unit_path, 'rb') as reader:

        # Load study metadata
        metadata = json.loads(next(reader))
        if only_meta:
            return metadata, None

        # Load sequencing data
        start = time.time()
        with parallel_backend('dask'):
            dfs = Parallel(n_jobs=n_jobs, verbose=100)(
                delayed(_parse_antibody_jsons)(
                    json_iterator=list(jsons),
                    keep_errors=keep_errors,
                    origin=str(oas_unit_path),
                    continue_on_error=continue_on_error,
                )
                for jsons in ichunked(reader, batch_size)
            )
        df = pd.concat(dfs).reset_index(drop=True)

        # Collect simple performance stats
        metadata['sequencing_parsing_n_jobs'] = n_jobs
        metadata['sequencing_parsing_batch_size'] = batch_size
        metadata['sequencing_parsing_taken_s'] = time.time() - start

        print(f'PARSED {len(df)} ANTIBODIES '
              f'IN {metadata["sequencing_parsing_taken_s"]:.2f} seconds '
              f'({Path(oas_unit_path).stem})')

        return metadata, df


class Unit:
    """
    The brainfuck is we need to deal with several versions of OAS and confusing terminology.

    In the original OAS terminology there are:
      - Unpaired units: the initial and larger part of the dataset
      - Paired units: later added, they also contain unpaired sequences (to add confusion)

    There are also two types of units:
      - JSON: the original ones, also have associated the original nucleotide data
              (not managed here)
      - CSV: a newer format
    """

    def __init__(self,
                 unit_id: str,
                 study_id: str = 'Banarjee_2017',
                 oas_path: Union[str, Path] = None):

        super().__init__()

        self._unit_id = unit_id
        self._study_id = study_id
        if oas_path is None:
            oas_path = find_oas_path()
        self._oas_path = Path(oas_path)

        # --- Sanity checks

        has_unpaired = self._unpaired_data_path is not None
        has_paired_paired = self._paired_paired_data_path is not None
        has_paired_unpaired = self._paired_unpaired_data_path is not None

        # Check there is at least one source of data
        if not has_unpaired and not (has_paired_unpaired or has_paired_paired):
            raise Exception(f'UNIT {self.study_id} {self.unit_id} has no data.')

        # Ensure we only have one kind (paired or unpaired).
        # We could deal with this gracefully too, if it ever happens to exist.
        if has_unpaired and (has_paired_unpaired or has_paired_paired):
            raise Exception(f'UNIT {self.study_id} {self.unit_id} '
                            f'has data in both the PAIRED and UNPAIRED OAS subsets.')

        # Ensure we have both paired and unpaired data for relevant units.
        # We could deal with this gracefully too, if it ever happens to exist.
        if has_paired_paired ^ has_paired_unpaired:
            raise Exception(f'UNIT {self.study_id} {self.unit_id} '
                            f'is in the PAIRED OAS subset but does not have both paired and unpaired data.')

    @property
    def unit_id(self):
        return self._unit_id

    @property
    def study_id(self):
        return self._study_id

    @property
    def oas_path(self):
        return self._oas_path

    @cached_property
    def _paired_paired_data_path(self) -> Optional[Path]:
        path = self.oas_path / 'paired' / 'csv' / self.study_id / f'{self.unit_id}_paired.csv.gz'
        if path.is_file():
            return path
        return None

    @cached_property
    def _paired_unpaired_data_path(self) -> Optional[Path]:
        path = self.oas_path / 'paired' / 'csv_unpaired' / self.study_id / f'{self.unit_id}_unpaired.csv.gz'
        if path.is_file():
            return path
        return None

    @cached_property
    def _unpaired_data_path(self) -> Optional[Path]:
        paths = []
        for unit_format in ('.csv.gz', '.json.gz'):
            path = self.oas_path / 'unpaired' / 'json' / self.study_id / f'{self.unit_id}{unit_format}'
            if path.is_file():
                paths.append(path)
        if len(paths) == 2:
            raise Exception(f'Found both CSV and JSON unit formats for unit {self.study_id} {self.unit_id}')
        if 1 == len(paths):
            return paths[0]
        return None

    @property
    def is_unpaired(self):
        """Returns True iff the unit is in the UNPAIRED OAS subset."""
        return self._unpaired_data_path is not None

    @staticmethod
    def _cache(unit_path: Path,
               recompute: bool = False,
               n_jobs: int = -1,
               batch_size: int = 10_000,
               continue_on_error: bool = True) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:

        cache_meta_path = unit_path.with_suffix('.meta.json')
        cache_data_path = unit_path.with_suffix('.parquet')

        if not recompute:
            try:
                with open(cache_meta_path, 'rt') as reader:
                    return json.load(reader), from_parquet(cache_data_path)
            except (IOError, ArrowInvalid):
                ...

        if unit_path.stem.endswith('.csv'):
            meta, df = parse_oas_csv_unit(oas_unit_path=unit_path)
        else:
            meta, df = parse_oas_json_unit(oas_unit_path=unit_path,
                                           only_meta=False,
                                           keep_errors=False,
                                           n_jobs=n_jobs,
                                           batch_size=batch_size,
                                           continue_on_error=continue_on_error)

        with open(cache_meta_path, 'wt') as writer:
            json.dump(meta, writer, indent=2)
        to_parquet(df, cache_data_path, compression='zstd', compression_level=20)

        return meta, df

    def paired(self,
               recompute=False,
               n_jobs=-1,
               batch_size=10_000,
               continue_on_error: bool = False) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        if self.is_unpaired:
            return None, None
        return self._cache(self._paired_paired_data_path,
                           recompute=recompute,
                           n_jobs=n_jobs,
                           batch_size=batch_size,
                           continue_on_error=continue_on_error)

    def unpaired(self,
                 recompute=False,
                 n_jobs=-1,
                 batch_size=10_000,
                 continue_on_error: bool = False) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
        path = self._unpaired_data_path if self.is_unpaired else self._paired_unpaired_data_path
        return self._cache(path,
                           recompute=recompute,
                           n_jobs=n_jobs,
                           batch_size=batch_size,
                           continue_on_error=continue_on_error)

    def smaller(self,
                size=1024,
                seed=42,
                dest_dir=RELATIVE_OAS_TEST_DATA_PATH):
        """Generate a subsample of the unit."""
        raise NotImplemented
        # if self.is_unpaired:
        #     metadata, df = AN_UNPAIRED_JSON_UNIT.unpaired()
        #     df = df.sample(n=size, random_state=seed, replace=False)
        #     to_parquet(df, Path.home() / 'test.parquet')


class Study:

    def __init__(self,
                 study_id: str = 'Banarjee_2017',
                 oas_path: Union[str, Path] = None):
        super().__init__()
        self._study_id = study_id
        if oas_path is None:
            oas_path = find_oas_path()
        self._oas_path = Path(oas_path)

    @property
    def oas_path(self):
        return self._oas_path

    @property
    def study_id(self):
        return self._study_id

    # --- Factories

    def units(self) -> Iterator[Unit]:
        all_unit_ids = sorted(set(chain(
            (unit_path.stem.split('_')[0] for unit_path in chain(
                # 10x genomics units with paired VH VL data
                (self.oas_path / 'paired' / 'csv' / self.study_id).glob('*.gz'),
                # 10x genomics units with unpaired VH or VL data
                (self.oas_path / 'paired' / 'csv_unpaired' / self.study_id).glob('*.gz')
            )),
            (unit_path.stem.replace('.csv', '').replace('.json', '') for unit_path in chain(
                # units with unpaired processed VH or VL data
                (self.oas_path / 'unpaired' / 'json' / self.study_id).glob('*.gz'),
                # N.B. we ignore unprocessed nucleotide data
            ))
        )))
        for unit_id in all_unit_ids:
            yield Unit(unit_id=unit_id, study_id=self.study_id, oas_path=self.oas_path)

    @classmethod
    def studies(cls, oas_path: Union[str, Path] = None) -> Iterator['Study']:
        if oas_path is None:
            oas_path = find_oas_path()
        oas_path = Path(oas_path)
        all_study_ids = sorted(set(
            study_path.stem for study_path in chain(
                # 10x genomics studies with paired VH VL data
                (oas_path / 'paired' / 'csv').glob('[A-Z]*'),
                # 10x genomics studies with unpaired VH or VL data
                (oas_path / 'paired' / 'csv_unpaired').glob('[A-Z]*'),
                # studies with unpaired processed VH or VL data
                (oas_path / 'unpaired' / 'json').glob('[A-Z]*'),
                # N.B. we ignore unprocessed nucleotide data
            )))
        for study_id in all_study_ids:
            yield Study(study_id=study_id, oas_path=oas_path)


# --- Pre-processing numberings
# TODO: maybe we could already save the token index too...
# TODO: likely we should re-annotate everything

LIGHT_REGIONS = 'fwl1', 'cdrl1', 'fwl2', 'cdrl2', 'fwl3', 'cdrl3', 'fwl4'
HEAVY_REGIONS = 'fwh1', 'cdrh1', 'fwh2', 'cdrh2', 'fwh3', 'cdrh3', 'fwh4'

ANARCI_IMGT_CDR_LENGTHS = {
    #
    # Notes:
    #
    #   - CDR1 and CDR2 lengths shall be imposed by germline and so not exceed their observed limits
    #     (e.g., IMGT bounds). But ANARCI sometimes reports longer CDRs
    #     (with the famous 62A bug, for example) and so we can flag problematic Abs (see explanations above)
    #
    #   - CDR3: IMGT really poses no constraints to the length of CDR3
    #       http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    #     But ANARCI claims it to be 65, and in turn to be able to recognize only shorter CDR3s.
    #     However, reading some code:
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/lib/python/anarci/schemes.py#L421-L436
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/build/lib/anarci/schemes.py#L485-L491
    #     Length seems limited by the insertions alphabet (52 insertion codes + space for "no insertion"):
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/build/lib/anarci/schemes.py#L82
    #       ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    #        "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "BB", "CC", "DD", "EE", "FF", "GG",
    #        "HH", "II", "JJ", "KK", "LL", "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU",
    #        "VV", "WW", "XX", "YY", "ZZ", " "]
    #     In practice we likely cannot at this stage detect unsupported CDR3 lengths, as these shall be filtered out
    #     by ANARCI / the data processing pipeline / nature.
    #     Still let's check.

    #
    # CDR length limits, both inclusive
    #

    'cdr1': (0, 12),  # N.B. ignore IMGT lower bound of 5
    'cdr2': (0, 10),
    'cdr3': (0, 106),  # N.B. ANARCI limit 53 insertion codes x 2
}


def preprocess_anarci_data(numbering_data_dict, expected_sequence=None, expected_cdr3=None) -> dict:
    """
    Parses the ANARCI imgt annotations in the original OAS units into a more efficient representation.
    Flags potential problems.
    """

    #
    # Explaining what follows.
    #
    # OAS was IMGT annotated with some (old?) version of ANARCI
    #   http://www.imgt.org/IMGTindex/numbering.php
    #
    # Theoretically IMGT insertions in CDR3 should only happen in residues 111 and 112
    # In 112 they follow an inverse order (i.e., B < A)
    # Like: 111, 111A, 111B, 112B, 112A, 112
    # See:
    #   https://github.com/oxpig/ANARCI/blob/master/lib/python/anarci/schemes.py
    #   http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    #   http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGTmutation.html
    #   http://www.imgt.org/FAQ/
    # In practice, we are dealing with the unexpected due to incomplete coverage
    # of existing Ab variants and bioinformatics / software problems.
    #
    # These are the known ANARCI bugs that might affect some IMGT positions:
    #
    #   - Heavy 61A before 61:
    #     https://github.com/oxpig/ANARCI/issues/14
    #     This seems actually a corner case when a CDR2 is longer than allowed by the scheme's 10 max length?
    #
    #   - Kappa 21 spurious gap (only newer ANARCIs):
    #     https://github.com/oxpig/ANARCI/issues/17
    #     We could bisect, or check one of several forks seemingly dealing with new germlines
    #
    # Plus other relevant ANARCI bugs:
    #   - Useless checking for too long CDRs
    #     https://github.com/oxpig/ANARCI/issues/8
    #     See also first bug above
    #
    # Some sanity checks stored as flags:
    #
    #   - insertion codes in positions other than 111/112 (e.g. 61A bug)
    #
    #   - too long CDRs (related to previous)
    #     Gabi's take:
    #       > Hmm... so, the V segments contain CDRs 1 and 2, so these shouldn't get longer
    #         than germline - are you looking at a different species antibody where you're seing this?
    #         If so, I suppose they would use the same .1 etc scheme that they use for CDR3, although
    #         they were probably assuming at the time that no longer loops existed...
    #         I think they included everything they knew at the time, but surely there are some odd ruminants
    #         that have freak antibodies.
    #         Are those datasets human? Could also be some experimental oddity, although I don't recall
    #         seeing changes in sequence length...
    #         If you can extract the sequences, you could see whether it's including some framework into the CDR
    #         - at least CDR1 is pretty easy to recognize
    #
    #   - not conserved cysteines
    #     Gabi's take:
    #       > first - yes, I think they mean absence of cysteins in IMGT positions 23 and/or 104.
    #         You probably can crete antibodies without these disulfides, and some people have successfully
    #         expressed antibodies in E. coli, which aren't able to form disulfide bonds, but typically
    #         those anitbodies aren't very good. In NGS data from an artificial library, I would absolutely
    #         kick sequences without either of the Cys, because it's most likely an artifact of some sort;
    #         if it's a natrual repertoire, it may be real (or a sequencing artifact, which isn't super rare in NGS),
    #         but I'd still rather not work with that antibody
    #     Note, we could extend the check to other conserved features. From the IMGT page:
    #        > The IMGT unique numbering has been defined to compare the variable domains whatever
    #          the antigen receptor, the chain type, or the species [1-3]. In the IMGT unique numbering,
    #          the conserved amino acids always have the same position, for instance cysteine 23 (1st-CYS),
    #          tryptophan 41 (CONSERVED-TRP), hydrophobic amino acid 89, cysteine 104 (2nd-CYS),
    #          phenylalanine or tryptophan 118 (J-PHE or J-TRP).
    #     But AFAICT this does not have support in the literature, as opposed to the 1st and 2nd cysteine checks
    #     See:
    #       https://doi.org/10.1101/2021.01.08.425894
    #       https://doi.org/10.4049/jimmunol.1800669
    #
    #   - wrong sequence reconstruction
    #
    #   - unfit (having any of the previous)
    #

    #
    # Slow, but since we run only once...
    #

    # Figure out if we are dealing with Light or Heavy chain
    has_heavy = any(region in numbering_data_dict for region in HEAVY_REGIONS)
    has_light = any(region in numbering_data_dict for region in LIGHT_REGIONS)
    if has_heavy and has_light:
        raise ValueError(f'Mixed heavy and light chain')
    if not has_heavy and not has_light:
        raise ValueError(f'No antibody regions found')
    regions = LIGHT_REGIONS if has_light else HEAVY_REGIONS

    # We will populate all these fields for the current record
    alignment_data = {
        # flags
        'unfit': False,
        'has_unexpected_insertions': False,
        'has_mutated_conserved_cysteines': False,
        'has_wrong_sequence_reconstruction': None,
        'has_wrong_cdr3_reconstruction': None,
        'has_long_cdr1': None,
        'has_long_cdr2': None,
        'has_long_cdr3': None,
        # alignment
        'domain': 'vl' if has_light else 'vh',
        'has_insertions': False,
        'fw1_start': None,
        'fw1_length': None,
        'cdr1_start': None,
        'cdr1_length': None,
        'fw2_start': None,
        'fw2_length': None,
        'cdr2_start': None,
        'fw3_start': None,
        'fw3_length': None,
        'cdr2_length': None,
        'cdr3_start': None,
        'cdr3_length': None,
        'fw4_start': None,
        'fw4_length': None,
        'aligned_sequence': [],
        'positions': [],
        'insertions': [],
    }

    last_region_end = 0
    for region in regions:

        # What AAs do we have in the region?
        aas_in_region = list(numbering_data_dict.get(region, {}).items())

        # Normalize region name (this will need to be undone for paired data in the same record)
        region = region.replace('l', '').replace('h', '')

        # Start and length (None and 0 for not present regions)
        alignment_data[f'{region}_start'] = last_region_end if len(aas_in_region) else None
        alignment_data[f'{region}_length'] = len(aas_in_region)
        last_region_end += len(aas_in_region)

        # Detect unsupported CDR lengths (we could do the same with framework regions)
        if region.startswith('cdr') and 0 < len(aas_in_region):
            alignment_data[f'has_long_{region}'] = len(aas_in_region) > ANARCI_IMGT_CDR_LENGTHS[region][1]

        # Sort the region AAs
        region_positions = []
        for position, aa in aas_in_region:
            try:
                position, insertion_code = int(position), ' '
            except ValueError:
                position, insertion_code = int(position[:-1]), position[-1]
                # Got insertions
                alignment_data['has_insertions'] = True
                if position not in (111, 112):
                    alignment_data['has_unexpected_insertions'] = True
            if position in (23, 104) and aa != 'C':
                alignment_data['has_mutated_conserved_cysteines'] = True
            # Mirroring of inserted residues
            insertion_code_order = ord(insertion_code) if position not in (112, 62) else -ord(insertion_code)
            region_positions.append((position, insertion_code_order, insertion_code, aa))
        region_positions = sorted(region_positions)

        alignment_data['aligned_sequence'] += [aa for *_, aa in region_positions]
        alignment_data['positions'] += [position for position, *_ in region_positions]
        alignment_data['insertions'] += [insertion for *_, insertion, _ in region_positions]

    # Make the alignment data a tad more efficient to work with. Still playing...
    #   - Likely using arrays for aligned sequences is not needed
    #     (and precludes parquet from better representation?)
    #     If so, just ''.join() both the sequence and the insertion codes.
    #   - Probably sparse insertion codes make more sense performancewise,
    #     they are less practical though.
    #   - Likely u1 dtype can work for positions (evaluate after collecting stats).
    alignment_data['aligned_sequence'] = np.array(alignment_data['aligned_sequence'], dtype='S1')
    alignment_data['positions'] = np.array(alignment_data['positions'], dtype=np.dtype('u2'))
    alignment_data['insertions'] = (np.array(alignment_data['insertions'], dtype='S2')
                                    if alignment_data['has_insertions'] else None)
    if expected_sequence is not None:
        alignment_data['has_wrong_sequence_reconstruction'] = (
                alignment_data['aligned_sequence'].tobytes().decode('utf-8') != expected_sequence
        )
    if expected_cdr3 is not None:
        if alignment_data['cdr3_start'] is None:
            alignment_data['has_wrong_cdr3_reconstruction'] = True
        else:
            cdr3_start = alignment_data['cdr3_start']
            cdr3_end = alignment_data['cdr3_start'] + alignment_data['cdr3_length']
            aligned_cdr3 = alignment_data['aligned_sequence'][cdr3_start:cdr3_end].tobytes().decode('utf-8')
            alignment_data['has_wrong_cdr3_reconstruction'] = aligned_cdr3 != expected_cdr3

    # Final veredict about the fitness of the chain
    QC_FLAGS = (
        'has_unexpected_insertions',
        'has_mutated_conserved_cysteines',
        'has_wrong_sequence_reconstruction',
        'has_wrong_cdr3_reconstruction',
        'has_long_cdr1',
        'has_long_cdr2',
        'has_long_cdr3',
    )
    alignment_data['unfit'] = any(alignment_data[flag] for flag in QC_FLAGS)

    return alignment_data


# --- Smoke tests (to be turned into actual tests)


AN_UNPAIRED_JSON_UNIT = Unit(unit_id='Cui_2016_immunize2_immunized_mouse_2_iglblastn', study_id='Cui_2016')
AN_UNPAIRED_CSV_UNIT = Unit(unit_id='SRR11610492_1_igblastn_anarci_Bulk', study_id='Nielsen_2020')
A_PAIRED_UNIT = Unit(unit_id='SRR11528761', study_id='Alsoiussi_2020')


def show_unit_data_example(unit: Unit, recompute=True, dask_client_address=None):
    from pprint import pprint

    _initialize_dask(dask_client_address=dask_client_address)

    unit_id = (f'{unit.study_id} {unit.unit_id} '
               f'{"(UNPAIRED_SET)" if unit.is_unpaired else "(PAIRED_SET)"}')

    # Paired
    meta, df = unit.paired(recompute=recompute)
    if meta is not None:
        print('-' * 80)
        print(f'PAIRED DATA FOR UNIT: {unit_id}')
        pprint(meta)
        pprint(df.iloc[0].to_dict())
        print('-' * 80)

    # Unpaired
    print('-' * 80)
    print(f'UNPAIRED DATA FOR UNIT: {unit_id}')
    meta, df = unit.unpaired(recompute=recompute)
    pprint(meta)
    pprint(df.iloc[0].to_dict())
    print('-' * 80)


def smoke_test_preprocess_anarco_data_unpaired_json(dask_client_address=None):

    UNPAIRED_JSON_ANTIBODY_EXAMPLE = {
        'cdr3': 'ALWYNNHWV',
        'data': {
            'cdrl1': {'27': 'T',
                      '28': 'G',
                      '29': 'A',
                      '30': 'V',
                      '31': 'T',
                      '35': 'T',
                      '36': 'S',
                      '37': 'N',
                      '38': 'Y'},
            'cdrl2': {'56': 'G', '57': 'T', '65': 'K'},
            'cdrl3': {'105': 'A',
                      '106': 'L',
                      '107': 'W',
                      '108': 'Y',
                      '109': 'N',
                      '114': 'N',
                      '115': 'H',
                      '116': 'W',
                      '117': 'V'},
            'fwl1': {'1': 'Q',
                     '11': 'L',
                     '12': 'T',
                     '13': 'T',
                     '14': 'S',
                     '15': 'P',
                     '16': 'G',
                     '17': 'E',
                     '18': 'T',
                     '19': 'V',
                     '2': 'A',
                     '20': 'T',
                     '21': 'V',
                     '22': 'T',
                     '23': 'C',
                     '24': 'R',
                     '25': 'S',
                     '26': 'S',
                     '3': 'V',
                     '4': 'V',
                     '5': 'P',
                     '6': 'E',
                     '7': 'E',
                     '8': 'S',
                     '9': 'A'},
            'fwl2': {'39': 'A',
                     '40': 'N',
                     '41': 'W',
                     '42': 'V',
                     '43': 'Q',
                     '44': 'E',
                     '45': 'K',
                     '46': 'P',
                     '47': 'D',
                     '48': 'H',
                     '49': 'L',
                     '50': 'F',
                     '51': 'T',
                     '52': 'G',
                     '53': 'L',
                     '54': 'I',
                     '55': 'G'},
            'fwl3': {'100': 'A',
                     '101': 'I',
                     '102': 'Y',
                     '103': 'F',
                     '104': 'C',
                     '66': 'N',
                     '67': 'R',
                     '68': 'A',
                     '69': 'P',
                     '70': 'G',
                     '71': 'V',
                     '72': 'P',
                     '74': 'A',
                     '75': 'R',
                     '76': 'F',
                     '77': 'S',
                     '78': 'G',
                     '79': 'S',
                     '80': 'L',
                     '83': 'I',
                     '84': 'G',
                     '85': 'D',
                     '86': 'K',
                     '87': 'A',
                     '88': 'A',
                     '89': 'L',
                     '90': 'T',
                     '91': 'I',
                     '92': 'T',
                     '93': 'G',
                     '94': 'A',
                     '95': 'Q',
                     '96': 'T',
                     '97': 'E',
                     '98': 'D',
                     '99': 'E'},
            'fwl4': {'118': 'F',
                     '119': 'G',
                     '120': 'G',
                     '121': 'G',
                     '122': 'T',
                     '123': 'K',
                     '124': 'L',
                     '125': 'T',
                     '126': 'V',
                     '127': 'L'}},
        'errors': [None],
        'j': 'IGLJ1*01',
        'name': 1,
        'num_errors': 0,
        'original_name': '21726',
        'redundancy': 1,
        'seq': 'QAVVPEESALTTSPGETVTVTCRSSTGAVTTSNYANWVQEKPDHLFTGLIGGTK'
               'NRAPGVPARFSGSLIGDKAALTITGAQTEDEAIYFCALWYNNHWVFGGGTKLTVL',
        'v': 'IGLV1*01'
    }

    _initialize_dask(dask_client_address=dask_client_address)

    print(preprocess_anarci_data(numbering_data_dict=UNPAIRED_JSON_ANTIBODY_EXAMPLE['data'],
                                 expected_sequence=UNPAIRED_JSON_ANTIBODY_EXAMPLE['seq'],
                                 expected_cdr3=UNPAIRED_JSON_ANTIBODY_EXAMPLE['cdr3']))


def smoke_test_units(dask_client_address=None):

    _initialize_dask(dask_client_address=dask_client_address)

    assert AN_UNPAIRED_JSON_UNIT.is_unpaired
    show_unit_data_example(AN_UNPAIRED_JSON_UNIT)

    assert AN_UNPAIRED_CSV_UNIT.is_unpaired
    show_unit_data_example(AN_UNPAIRED_CSV_UNIT)

    assert not A_PAIRED_UNIT.is_unpaired
    show_unit_data_example(A_PAIRED_UNIT)


def smoke_test_access_all(dask_client_address=None):

    _initialize_dask(dask_client_address=dask_client_address)

    for study in Study.studies():
        for unit in study.units():
            print(unit.study_id, unit.unit_id)
            unit.paired()
            unit.unpaired()


def smoke_test_parallel_json_preprocessing(json_unit=AN_UNPAIRED_JSON_UNIT,
                                           n_jobss=(1, 2, 4, 8),
                                           batch_sizes=(10_000, 20_000),
                                           dask_client_address=None):

    _initialize_dask(dask_client_address=dask_client_address)

    # FIXME: beware, cheating benchmark, we need to drop file system caches before
    records = []
    expected_df = None
    for n_jobs, batch_size in product(n_jobss, batch_sizes):
        if expected_df is None:
            metadata, expected_df = json_unit.unpaired(recompute=True, n_jobs=n_jobs, batch_size=batch_size)
        else:
            metadata, df = json_unit.unpaired(recompute=True, n_jobs=n_jobs, batch_size=batch_size)
            pd.testing.assert_frame_equal(expected_df, df)
        records.append({
            'n_jobs': metadata['sequencing_parsing_n_jobs'],
            'batch_size': metadata['sequencing_parsing_batch_size'],
            'sequencing_parsing_taken_s': metadata['sequencing_parsing_taken_s'],
            'unit': json_unit.unit_id,
        })
    benchmark_df = pd.DataFrame(records)
    print(benchmark_df)
    return benchmark_df


# --- Actual processing commands


def _parse_json_units_for_study(*,
                                study,
                                recompute,
                                continue_on_error,
                                n_parsing_jobs,
                                batch_size,
                                dask_client_address=None):

    _initialize_dask(dask_client_address=dask_client_address)

    print(f'STUDY {study.study_id}')
    for unit in study.units():
        # noinspection PyProtectedMember
        if unit.is_unpaired and str(unit._unpaired_data_path).endswith('.json.gz'):
            print(f'UNIT {unit.unit_id}')
            unit.unpaired(recompute=recompute,
                          n_jobs=n_parsing_jobs,
                          batch_size=batch_size,
                          continue_on_error=continue_on_error)
            print(f'DONE UNIT {unit.unit_id}')
    print(f'DONE STUDY {study.study_id}')


def preprocess_json_units(recompute=False,
                          continue_on_error=False,
                          n_unit_jobs=8,
                          n_parsing_jobs=8,
                          batch_size=20_000):

    # Parallelize across units (parallel I/O)
    # Then parallelize across antibodies (parallel CPU)

    with parallel_backend('dask'):
        Parallel(n_jobs=n_unit_jobs, verbose=100)(
            delayed(_parse_json_units_for_study)(
                study=study,
                recompute=recompute,
                continue_on_error=continue_on_error,
                n_parsing_jobs=n_parsing_jobs,
                batch_size=batch_size,
            )
            for study in Study.studies()
        )


def main():
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        preprocess_json_units,
        smoke_test_preprocess_anarco_data_unpaired_json,
        smoke_test_units,
        smoke_test_access_all,
        smoke_test_parallel_json_preprocessing,
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()

# FIXME: current dask experiment has gone bad (much slower than old parallelization)
