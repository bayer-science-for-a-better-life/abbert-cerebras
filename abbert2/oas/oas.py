"""
Observed Antibody Space data manipulation.

OAS is licensed under CC-BY 4.0:
https://creativecommons.org/licenses/by/4.0/

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

UPDATE 2021/09
  - A new manuscript is in preparation in Dean's group.
  - Old download links have died in favor of a consolidated CSV-only data format.
"""
import json
import shutil
from builtins import IOError
from functools import cached_property, total_ordering
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, Union, Iterator, Optional, List, Callable

import pandas as pd
from pyarrow import ArrowInvalid
import pyarrow.parquet as pq

from abbert2.common import to_json_friendly, from_parquet, mtime, to_parquet
from abbert2.oas.common import find_oas_path, check_oas_subset


# --- Some field normalization code

def _normalize_oas_species(species):
    return {
        'camel': 'camel',

        'mouse': 'mouse',
        'mouse_C57BL/6': 'mouse',
        'mouse_BALB/c': 'mouse',
        'HIS-Mouse': 'mouse',
        'mouse_RAG2-GFP/129Sve': 'mouse',
        'mouse_Swiss-Webster': 'mouse',

        'human': 'human',

        'rabbit': 'rabbit',

        'rat_SD': 'rat',
        'rat': 'rat',

        'rhesus': 'rhesus',
    }.get(species, species)


# --- Convenient abstractions over the dataset


class OAS:
    """Top level management of OAS data."""

    def __init__(self, oas_path: Union[str, Path] = None):
        super().__init__()
        if oas_path is None:
            oas_path = find_oas_path()
        
        self._oas_path = Path(oas_path)
        # Added by aarti-cerebras
        if self._oas_path.is_file():
            *oas_path_dir, oas_subset, study_id, unit_id, _ = self._oas_path.parts
        else:
            *oas_path_dir, oas_subset, study_id, unit_id = self._oas_path.parts

        self._oas_path = Path(*oas_path_dir)
        self._oas_subset = oas_subset
        self._study_id = study_id
        self._unit_id = unit_id

    @property
    def oas_path(self) -> Path:
        return self._oas_path

    @property
    def oas_subset(self) -> str:
        # Added by aarti-cerebras
        return self._oas_subset

    @property
    def study_id(self) -> str:
        # Added by aarti-cerebras
        return self._study_id

    @property
    def unit_id(self) -> str:
        # Added by aarti-cerebras
        return self._unit_id

    @cached_property
    def unit_metadata_df(self):
        from abbert2.oas.preprocessing import oas_units_meta
        return oas_units_meta(oas_path=self.oas_path,
                              paired=None,
                              keep_missing=False)

    def unit_metadata(self, oas_subset: str, study_id: str, unit_id: str) -> dict:
        df = self.unit_metadata_df
        df = df.query(f'oas_subset == "{oas_subset}" and study_id == "{study_id}" and unit_id == "{unit_id}"')
        if len(df) > 1:
            raise Exception(f'Ambiguous metadata for unit ({oas_subset}, {study_id}, {unit_id})')
        if 0 == len(df):
            raise Exception(f'Cannot find metadata for unit ({oas_subset}, {study_id}, {unit_id})')
        return df.iloc[0].to_dict()

    def populate_metadata_jsons(self):
        # TODO: force update + flagging of changed or removed units (usual missing update lifecycle everywhere)
        for unit in self.units_in_meta():
            _ = unit.metadata  # Side effects FTW

    # --- Factories

    def unit(self, oas_subset: str, study_id: str, unit_id: str) -> 'Unit':
        return Unit(oas_subset=oas_subset,
                    study_id=study_id,
                    unit_id=unit_id,
                    oas_path=self.oas_path,
                    oas=self)

    def unit_from_path(self, path: Union[str, Path]) -> 'Unit':
        path = Path(path)
        if path.is_file():
            *_, oas_subset, study_id, unit_id, _ = path.parts
        else:
            *_, oas_subset, study_id, unit_id = path.parts
        return self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)

    def units_in_disk(self, oas_subset: str = None) -> Iterator['Unit']:
        if oas_subset is None:
            yield from chain(self.units_in_disk(oas_subset='paired'), self.units_in_disk(oas_subset='unpaired'))
        else:
            check_oas_subset(oas_subset)
            for study_path in sorted((self.oas_path / oas_subset).glob('*')):
                if study_path.is_dir():
                    for unit_path in study_path.glob('*'):
                        if unit_path.is_dir():
                            yield self.unit(oas_subset, study_path.stem, unit_path.stem)


    def units_in_path(self):
        # Added by aarti-cerebras
        check_oas_subset(self.oas_subset)

        unit_path = (self.oas_path / self.oas_subset / self.study_id / self.unit_id)
        if unit_path.is_dir():
            # print(f"-----units_in_new study_path {self.oas_subset}, {self.study_id}, {unit_path.stem}")
            yield self.unit(self.oas_subset, self.study_id, unit_path.stem)

    def units_in_meta(self) -> Iterator['Unit']:
        df = self.unit_metadata_df
        for oas_subset, study_id, unit_id in zip(df['oas_subset'], df['study_id'], df['unit_id']):
            yield self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)

    # --- Caches

    def _add_units_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'unit' in df.columns:
            raise Exception('Dataframe already has a unit column')
        df['unit'] = [self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)
                      for oas_subset, study_id, unit_id in
                      zip(df['oas_subset'], df['study_id'], df['unit_id'])]
        return df

    def nice_unit_meta_df(self,
                          recompute: bool = False,
                          normalize_species: bool = True) -> pd.DataFrame:

        cache_path = self.oas_path / self.oas_subset / self.study_id / self.unit_id / 'nice_unit_meta_df.parquet'

        df = None

        if not recompute:
            try:
                df = from_parquet(cache_path)
            except (IOError, FileNotFoundError):
                ...

        if df is None:
            # Added by aarti-cerebras
            # df = pd.DataFrame([unit.nice_metadata for unit in self.units_in_disk()])
            df = pd.DataFrame([unit.nice_metadata for unit in self.units_in_path()])
            to_parquet(df, cache_path)

        # add units for full access to the data
        self._add_units_to_df(df)

        if normalize_species:
            df['species'] = df['species'].apply(_normalize_oas_species)

        # TODO: ask everyone to update the cache and remove this
        #       study_year was buggy...
        df['study_year'] = df['unit'].apply(lambda unit: unit.study_year)

        for int_column in ('online_csv_size_bytes',
                           'study_year',
                           'theoretical_num_sequences_unique',
                           'theoretical_num_sequences_total',
                           'sequences_file_size',
                           'sequences_num_records',
                           'heavy_cdr3_max_length'):
            df[int_column] = df[int_column].astype(pd.Int64Dtype())

        return df


@total_ordering
class Unit:
    """Manage a single OAS unit."""

    def __init__(self,
                 *,
                 oas_subset: str,
                 study_id: str,
                 unit_id: str,
                 oas_path: Union[str, Path] = None,
                 oas: OAS = None):
        super().__init__()
        self._oas_subset = oas_subset
        self._study_id = study_id
        self._unit_id = unit_id
        if oas_path is None:
            oas_path = find_oas_path()
        self._oas_path = Path(oas_path)
        self._oas = oas

    # --- Unit coordinates

    @property
    def id(self) -> Tuple[str, str, str]:
        return self.oas_subset, self.study_id, self.unit_id

    @property
    def oas_subset(self) -> str:
        return self._oas_subset

    @property
    def study_id(self) -> str:
        return self._study_id

    @property
    def unit_id(self) -> str:
        return self._unit_id

    @property
    def oas_path(self) -> Path:
        return self._oas_path

    @property
    def path(self) -> Path:
        return self.oas_path / self.oas_subset / self.study_id / self.unit_id

    @property
    def oas(self) -> OAS:
        if self._oas is None:
            self._oas = OAS(self.oas_path)
        return self._oas

    # --- Original CSV.gz file

    @property
    def original_csv_path(self) -> Path:
        return self.path / f'{self.unit_id}.csv.gz'

    @property
    def has_original_csv(self) -> bool:
        return self.original_csv_path.is_file()

    @property
    def original_local_csv_mdate(self) -> Optional[pd.Timestamp]:
        try:
            return mtime(self.original_csv_path)
        except FileNotFoundError:
            return None

    @property
    def needs_redownload(self):
        if not self.has_original_csv:
            return True
        is_old = self.original_local_csv_mdate < self.online_modified_date
        is_different_size = self.original_csv_path.stat().st_size != self.online_csv_size_bytes
        return is_old or is_different_size

    # --- Unit metadata

    @property
    def metadata_path(self) -> Path:
        return self.path / f'{self.unit_id}.metadata.json'

    @cached_property
    def metadata(self):
        try:
            with self.metadata_path.open('rt') as reader:
                return json.load(reader)
        except (FileNotFoundError, IOError, JSONDecodeError):
            metadata = self.oas.unit_metadata(oas_subset=self.oas_subset,
                                              study_id=self.study_id,
                                              unit_id=self.unit_id)
            metadata = {k: to_json_friendly(v) for k, v in metadata.items()}
            self.persist_metadata(metadata)
            return metadata

    @cached_property
    def nice_metadata(self):
        fields = (
            'oas_subset', 'study_id', 'unit_id',
            'download_date', 'online_modified_date',
            'online_csv_size_bytes',
            'sequencing_run',
            'publication_link', 'study_author', 'study_year',
            'species', 'age',
            'bsource', 'btype',
            'subject', 'disease', 'vaccine',
            'longitudinal',
            'chain', 'isotype',
            'theoretical_num_sequences_unique', 'theoretical_num_sequences_total',
            'original_url', 'has_original_csv', 'original_local_csv_mdate', 'download_error', 'needs_redownload',
            'sequences_file_size', 'has_broken_sequences_file', 'sequences_num_records', 'sequences_miss_processing',
            'has_heavy_sequences', 'has_light_sequences', 'heavy_cdr3_max_length'
        )
        return {field: getattr(self, field) for field in fields}

    def persist_metadata(self, metadata=None):
        if metadata is None:
            metadata = self.metadata  # Beware infinite recursion
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open('wt') as writer:
            json.dump(metadata, writer, indent=2)

    @property
    def original_url(self) -> Optional[str]:
        return self.metadata.get('url')

    @property
    def download_error(self) -> Optional[str]:
        return self.metadata.get('http_error')

    @property
    def download_date(self) -> Optional[pd.Timestamp]:
        return pd.to_datetime(self.metadata.get('Date'))

    @property
    def online_modified_date(self) -> Optional[pd.Timestamp]:
        td: Optional[pd.Timestamp] = pd.to_datetime(self.metadata.get('Last-Modified'))
        if td is not None:
            # Workaround: on download file dates get truncated to the minute
            # This also truncates to the minute the online modified date, so comparisons hold
            return td.floor('min')
        return None

    @property
    def online_csv_size_bytes(self) -> Optional[int]:
        try:
            return int(self.metadata.get('Content-Length'))
        except TypeError:
            return None

    @property
    def sequencing_run(self) -> Optional[str]:
        return self.metadata.get('Run')

    @property
    def publication_link(self) -> Optional[str]:
        return self.metadata.get('Link')

    @property
    def study_author(self) -> Optional[str]:
        return self.metadata.get('Author')

    @property
    def study_year(self) -> int:
        return int(self.study_author.strip()[-4:])

    @property
    def species(self) -> Optional[str]:
        return self.metadata.get('Species')

    @property
    def normalized_species(self) -> Optional[str]:
        return _normalize_oas_species(self.species)

    @property
    def age(self) -> Optional[str]:
        # This probably we should clean and make it comparable
        return self.metadata.get('Age')

    @property
    def bsource(self) -> Optional[str]:
        return self.metadata.get('BSource')

    @property
    def btype(self) -> Optional[str]:
        return self.metadata.get('BType')

    @property
    def subject(self) -> Optional[str]:
        return self.metadata.get('Subject')

    @property
    def disease(self) -> Optional[str]:
        return self.metadata.get('Disease')

    @property
    def vaccine(self) -> Optional[str]:
        return self.metadata.get('Vaccine')

    @property
    def longitudinal(self) -> Optional[str]:
        return self.metadata.get('Longitudinal')

    @property
    def chain(self) -> Optional[str]:
        return self.metadata.get('Chain')

    @property
    def isotype(self) -> Optional[str]:
        return self.metadata.get('Isotype')

    @property
    def theoretical_num_sequences_unique(self) -> Optional[int]:
        try:
            return int(self.metadata.get('Unique sequences'))
        except TypeError:
            return None

    @property
    def theoretical_num_sequences_total(self) -> Optional[int]:
        try:
            return int(self.metadata.get('Total sequences'))
        except TypeError:
            return None

    @property
    def original_sequences_column_names(self) -> Optional[List[str]]:
        return self.metadata.get('column_names')

    # --- Sequences

    @property
    def processing_logs_file(self) -> Path:
        return self.path / f'{self.unit_id}.sequences.processing-logs.pickle'

    @property
    def has_processing_logs(self):
        return self.processing_logs_file.is_file()

    def processing_logs(self) -> Optional[dict]:
        if self.has_processing_logs:
            return pd.read_pickle(self.processing_logs_file)
        return None

    @property
    def processing_error_logs_file(self):
        return self.path / f'{self.unit_id}.sequences.processing-error.pickle'

    @property
    def has_processing_errors_log(self):
        return self.processing_error_logs_file.is_file()

    def processing_errors_log(self) -> Optional[dict]:
        if self.has_processing_errors_log:
            return pd.read_pickle(self.processing_error_logs_file)
        return None

    @property
    def sequences_path(self) -> Path:
        return self.path / f'{self.unit_id}.sequences.parquet'

    @property
    def has_sequences(self) -> bool:
        return self.sequences_path.is_file() and self._pq() is not None

    @property
    def has_broken_sequences_file(self) -> bool:
        return self.sequences_path.is_file() and self._pq() is None

    def should_recompute(self, force=False) -> bool:
        return (force or not self.has_sequences) and self.has_original_csv

    def sequences_df(self, columns=None) -> Optional[pd.DataFrame]:
        try:
            return from_parquet(self.sequences_path, columns=columns)
        except (IOError, FileNotFoundError, ArrowInvalid):
            return None

    @property
    def sequences_file_size(self) -> Optional[int]:
        if self.has_sequences:
            return self.sequences_path.stat().st_size
        return None

    def _pq(self) -> Optional[pq.ParquetFile]:
        try:
            return pq.ParquetFile(self.sequences_path)
        except (FileNotFoundError, IOError, ArrowInvalid):
            return None

    @property
    def has_heavy_sequences(self) -> bool:
        pq = self._pq()
        if pq is not None:
            return -1 != self._pq().schema_arrow.get_field_index('aligned_sequence_heavy')
        return False

    @property
    def has_light_sequences(self) -> bool:
        pq = self._pq()
        if pq is not None:
            return -1 != self._pq().schema_arrow.get_field_index('aligned_sequence_light')
        return False

    @property
    def sequences_num_records(self) -> Optional[int]:
        if self.has_sequences:
            return self._pq().metadata.num_rows
        return None

    @property
    def sequences_miss_processing(self) -> bool:
        num_records = self.sequences_num_records
        if num_records is None:
            return True
        # we could also check sum(redundancy) == theoretical_num_sequences_total...
        return num_records < self.theoretical_num_sequences_unique

    def region_max_length(self, region='cdr3', chain='heavy') -> Optional[int]:
        pq = self._pq()
        if pq is not None:
            column_index = pq.schema_arrow.get_field_index(f'{region}_length_{chain}')
            if column_index != -1:
                return max(pq.metadata.row_group(row_group).column(column_index).statistics.max
                           for row_group in range(pq.metadata.num_row_groups))
        return None

    @property
    def heavy_cdr1_max_length(self):
        return self.region_max_length(region='cdr1', chain='heavy')

    @property
    def heavy_cdr2_max_length(self):
        return self.region_max_length(region='cdr2', chain='heavy')

    @property
    def heavy_cdr3_max_length(self):
        return self.region_max_length(region='cdr3', chain='heavy')

    @property
    def light_cdr1_max_length(self):
        return self.region_max_length(region='cdr1', chain='light')

    @property
    def light_cdr2_max_length(self):
        return self.region_max_length(region='cdr2', chain='light')

    @property
    def light_cdr3_max_length(self):
        return self.region_max_length(region='cdr3', chain='light')

    # TODO: max sequence length

    # --- Magics

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Unit):
            raise ValueError(f'expecting Unit, got {type(other)}')
        return self.id == other.id

    def __lt__(self, other):
        if not isinstance(other, Unit):
            raise ValueError(f'expecting Unit, got {type(other)}')
        return self.id < other.id

    # --- More file management

    def copy_to(self,
                oas_path: Union[str, Path],
                include_sequences: bool = True,
                max_num_sequences: Optional[int] = None,
                include_original_csv: bool = False,
                overwrite: bool = False):

        oas_path = Path(oas_path)
        if oas_path == self.oas_path:
            raise Exception('Copying a unit over itself is not supported')

        dest_path = oas_path / self.oas_subset / self.study_id / self.unit_id

        def copy_but_do_not_overwrite(src):
            if not src.is_file():
                return
            dest = dest_path / src.name
            if dest.is_file() and not overwrite:
                raise Exception(f'Path already exists and will not overwrite ({dest})')
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest)

        # copy metadata
        copy_but_do_not_overwrite(self.metadata_path)

        # copy processed sequences
        if include_sequences and self.has_sequences:
            if max_num_sequences is None:
                copy_but_do_not_overwrite(self.sequences_path)
            else:
                dest = dest_path / self.sequences_path.name
                if dest.is_file() and not overwrite:
                    raise Exception(f'Path already exists and will not overwrite ({dest})')
                df = self.sequences_df()
                df = df.sample(n=min(max_num_sequences, len(df)), random_state=19)
                to_parquet(df, dest)
        if include_sequences:
            copy_but_do_not_overwrite(self.processing_logs_file)
            copy_but_do_not_overwrite(self.processing_error_logs_file)

        # copy original csv
        if include_original_csv:
            copy_but_do_not_overwrite(self.original_csv_path)


# --- Entry points


def populate_metadata_jsons(oas_path: Path = None):
    oas = OAS(oas_path=oas_path)
    oas.populate_metadata_jsons()


def extract_processed_oas(oas_path: Optional[Union[str, Path]] = None,
                          dest_path: Path = Path.home() / 'oas-processed',
                          overwrite: bool = False):
    oas = OAS(oas_path=oas_path)
    for unit in oas.units_in_disk():
        if unit.has_sequences:
            print(f'COPYING {unit.id}')
            unit.copy_to(dest_path,
                         include_sequences=True,
                         max_num_sequences=None,
                         include_original_csv=False,
                         overwrite=overwrite)
    print(f'Find your OAS dump in {dest_path}')


# --- Data partitioning and filtering examples

def sapiens_like_train_val_test(oas_path: Union[str, Path] = None) -> dict:
    """
    From: https://www.biorxiv.org/content/10.1101/2021.08.08.455394v1.full

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
    """

    # Let's maybe select using a dataframe, instead of looping through the units
    units_df = OAS(oas_path).nice_unit_meta_df(normalize_species=True, recompute=True)

    #
    # Sapiens sought humanization, so they built their model against human antibodies only
    # For us, it is not clear if this would be beneficial, but for the sake of reproduction
    # we filter out any non-human sequence.
    #
    # Would it make sense to add an auxiliary task to classify the sequence organism?
    #

    train_test_validation_dfs = {}

    for chain in ('heavy', 'light'):
        # print(f"---- human_units_df_species: {units_df.species}")
        # print(f"----- human_units_df_has_chain: , {units_df[f'has_{chain}_sequences']}")
        human_units_df = units_df.query(f'species == "human" and has_{chain}_sequences')
        
        train_test_validation_dfs[chain] = {
            # 'train': human_units_df.query('study_year <= 2017'),
            # 'validation': human_units_df.query('study_year == 2018'),
            # 'test': human_units_df.query('study_year >= 2019'),

            'train': human_units_df[human_units_df['study_year'] <= 2017],
            'validation': human_units_df[human_units_df['study_year'] == 2018],
            'test': human_units_df[human_units_df['study_year'] >= 2019],
        }

    return train_test_validation_dfs


def humab_like_filtering(sequences_df: pd.DataFrame,
                         chain: str = 'heavy') -> pd.DataFrame:
    """
    Filtering example ala Hu-mAb
    From:
     https://www.biorxiv.org/content/10.1101/2021.01.08.425894v2.full

    All IgG VH and VL sequences were downloaded from the OAS database (August 2020),
    totaling over 500 million sequences in the IMGT format. Human sequences were
    split by their V gene type - for example, V1 to V7 for VH sequences.
    Redundant sequences, sequences with cysteine errors (18) and sequences
    with missing framework 1 residues (residues preceding CDR1) were removed.
    The total dataset included over 65 million non-redundant sequences (SI section 1A).
    """

    # Filter out sequences without fw1
    sequences_df = sequences_df.query(f'fw1_length_{chain} > 0')

    # Filter out sequences with mutations in conserved cysteines
    if chain == 'heavy':
        has_mutated_conserved_cysteines = sequences_df[f'has_mutated_conserved_cysteines_{chain}'].apply(
            lambda x: x is not None and x  # account for both False and missing (but need to revisit the missing case)
        )
        sequences_df = sequences_df[~has_mutated_conserved_cysteines]
    # TODO: check aboss / anarci annotations and original paper to make sure this is correct
    # https://www.jimmunol.org/content/201/12/3694?ijkey=24817c8d879730cb4a170e371cfadd768703b0ed&keytype2=tf_ipsecsha

    return sequences_df


def train_validation_test_iterator(
        partitioner: Callable[[], dict] = sapiens_like_train_val_test,
        filtering: Optional[Callable[[pd.DataFrame, str], pd.DataFrame]] = humab_like_filtering,
        chains: Tuple[str, ...] = ('heavy', 'light'),
        ml_subsets: Tuple[str, ...] = ('train', 'validation', 'test'),
) -> Iterator[Tuple[Unit, str, str, pd.DataFrame]]:

    partition = partitioner()

    for chain in chains:
        used_qa_columns = [
            # Quality control
            f'has_mutated_conserved_cysteines_{chain}',
        ]
        used_ml_columns = [
            # To learn over
            f'fw1_start_{chain}',
            f'fw1_length_{chain}',
            f'cdr1_start_{chain}',
            f'cdr1_length_{chain}',
            f'fw2_start_{chain}',
            f'fw2_length_{chain}',
            f'cdr2_start_{chain}',
            f'cdr2_length_{chain}',
            f'fw3_start_{chain}',
            f'fw3_length_{chain}',
            f'cdr3_start_{chain}',
            f'cdr3_length_{chain}',
            f'fw4_start_{chain}',
            f'fw4_length_{chain}',
            f'aligned_sequence_{chain}',
        ]

        for ml_subset in ml_subsets:
            unit: Unit
            for unit in partition[chain][ml_subset]['unit']:
                unit_sequences_df = unit.sequences_df(columns=used_ml_columns + used_qa_columns)
                
                if unit_sequences_df is None:
                    yield unit, None, None, pd.DataFrame()
                    continue  # FIXME: this happens when the parquet file is broken beyond the schema
                try:
                    if filtering is not None:
                        unit_sequences_df = filtering(unit_sequences_df, chain)
                    # Drop QA columns
                    unit_sequences_df = unit_sequences_df.drop(columns=used_qa_columns)
                    yield unit, chain, ml_subset, unit_sequences_df
                except KeyError:
                    # FIXME: this happens when "has_mutated_conserved_cysteines_light" does not exist
                    #        observed in one unit, to troubleshoot
                    ...

# --- Smoke testing


if __name__ == '__main__':

    TEST_PAIRED_UNIT = Unit(oas_subset='paired',
                            study_id='Alsoiussi_2020',
                            unit_id='SRR11528761_paired',
                            oas_path=None)

    TEST_UNPAIRED_UNIT = Unit(oas_subset='unpaired',
                              study_id='Greiff_2017',
                              unit_id='ERR1759628_Heavy_Bulk',
                              oas_path=None)

    TEST_UNITS = TEST_PAIRED_UNIT, TEST_UNPAIRED_UNIT

    oas = OAS()
    oas.populate_metadata_jsons()

    oas.nice_unit_meta_df(recompute=False, normalize_species=True).info()

    for unit in oas.units_in_disk():
        print(unit.metadata)
        print(unit.nice_metadata)
        if unit.has_sequences:
            unit.sequences_df().info()
            unit.copy_to(Path.home() / 'small-oas-deleteme', max_num_sequences=100, overwrite=True)
        assert unit == unit
