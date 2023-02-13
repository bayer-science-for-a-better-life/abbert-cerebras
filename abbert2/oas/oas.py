"""
Observed Antibody Space data manipulation.

OAS is licensed under CC-BY 4.0:
https://creativecommons.org/licenses/by/4.0/

OAS home:
  http://opig.stats.ox.ac.uk/webapps/oas/
  (equivalently see antibodymap.org)

OAS papers:
  https://www.jimmunol.org/content/201/8/2502
  https://doi.org/10.1002/pro.4205

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
import time
from builtins import IOError
from collections import defaultdict
from functools import cached_property, total_ordering
from itertools import chain, zip_longest, islice, combinations
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, Union, Iterator, Optional, List, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import xxhash
from joblib import Parallel, delayed
from pyarrow import ArrowInvalid
from pyarrow import Table
from smart_open import open
from tqdm import tqdm

from abbert2.common import to_json_friendly, from_parquet, mtime, to_parquet, parse_anarci_position_aa_to_imgt_code, \
    anarci_imgt_code_to_insertion, parse_anarci_position_to_imgt_code
from abbert2.oas.common import find_oas_path, check_oas_subset, compress_sequences_df


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


# --- Misc utils

def copy_but_do_not_overwrite(src, dest_path, *, num_rows_header=-1, overwrite=False, fail_if_exists=False):
    if not src.is_file():
        return
    dest = dest_path / src.name
    if dest.is_file() and not overwrite:
        if fail_if_exists:
            raise Exception(f'Path already exists and will not overwrite ({dest})')
        return
    dest_path.mkdir(parents=True, exist_ok=True)
    if num_rows_header < 0:
        shutil.copy(src, dest)
    else:
        # this should be used just for the CSV
        with open(src, 'rt') as reader:
            with open(dest, 'wt') as writer:
                for line in islice(reader, num_rows_header):
                    writer.write(line)


# --- Convenient abstractions over the dataset


class OAS:
    """Top level management of OAS data."""

    def __init__(self, oas_path: Union[str, Path] = None):
        super().__init__()
        if oas_path is None:
            oas_path = find_oas_path()
        self._oas_path = Path(oas_path)

    @property
    def oas_path(self) -> Path:
        """Returns the path to the OAS dataset."""
        return self._oas_path

    @cached_property
    def unit_metadata_df(self) -> pd.DataFrame:
        """
        Returns the metadata collected for all the units collected from the internet.

        This is a wrapper over `preprocessing.oas_units_meta`.
        """
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

    def populate_metadata_jsons(self, recompute: bool = False):
        for unit in self.units_in_meta():
            if recompute:
                unit.update_metadata()
            else:
                _ = unit.metadata  # side effects FTW

    # --- Factories

    def unit(self, oas_subset: str, study_id: str, unit_id: str) -> 'Unit':
        """Returns a unit for the given coordinates."""
        return Unit(oas_subset=oas_subset,
                    study_id=study_id,
                    unit_id=unit_id,
                    oas_path=self.oas_path,
                    oas=self)

    def unit_from_path(self, path: Union[str, Path]) -> 'Unit':
        """Makes a best effort to return a Unit given a path."""
        path = Path(path)
        if path.is_file():
            *_, oas_subset, study_id, unit_id, _ = path.parts
        else:
            *_, oas_subset, study_id, unit_id = path.parts
        return self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)

    def units_in_disk(self, oas_subset: str = None) -> Iterator['Unit']:
        """
        Returns an iterator of units in disk.

         A unit in disk is defined by the existence of a directory
           oas_path/oas_subset/study_path/unit_path
         No further checks are carried.

        Parameters
        ----------
        oas_subset : "paired", "unpaired" or None
          Which OAS subset to iterate. If None, iterate in order "paired" and "unpaired" subsets
        """
        if oas_subset is None:
            yield from chain(self.units_in_disk(oas_subset='paired'), self.units_in_disk(oas_subset='unpaired'))
        else:
            check_oas_subset(oas_subset)
            for study_path in sorted((self.oas_path / oas_subset).glob('*')):
                if study_path.is_dir():
                    for unit_path in sorted(study_path.glob('*')):
                        if unit_path.is_dir():
                            yield self.unit(oas_subset, study_path.stem, unit_path.stem)

    def units_in_meta(self) -> Iterator['Unit']:
        """Returns an iterator of units in the collected OAS metadata."""
        df = self.unit_metadata_df
        for oas_subset, study_id, unit_id in zip(df['oas_subset'], df['study_id'], df['unit_id']):
            yield self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)

    def remove_units_not_in_meta(self, dry_run: bool = True):
        """Removes all the units in disk that are not in meta."""
        paths_in_disk = set(unit.path for unit in self.units_in_disk())
        paths_in_meta = set(unit.path for unit in self.units_in_meta())
        paths_to_remove = paths_in_disk - paths_in_meta
        if paths_to_remove:
            print(f'Removing {len(paths_to_remove)} units not present in meta')
            for path in sorted(paths_to_remove):
                print(f'Removing {path}')
                if not dry_run:
                    shutil.rmtree(path, ignore_errors=False)

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
                          normalize_species: bool = True,
                          add_units: bool = True) -> pd.DataFrame:

        cache_path = self.oas_path / 'summaries' / 'nice_unit_meta_df.parquet'

        df = None

        if not recompute:
            try:
                df = from_parquet(cache_path)
            except (IOError, FileNotFoundError):
                ...

        if df is None:
            df = pd.DataFrame([unit.nice_metadata for unit in self.units_in_disk()])
            to_parquet(df, cache_path)

        # add units for full access to the data
        if add_units:
            self._add_units_to_df(df)

        if normalize_species:
            df['species'] = df['species'].apply(_normalize_oas_species)

        for int_column in ('online_csv_size_bytes',
                           'study_year',
                           'theoretical_num_sequences_unique',
                           'theoretical_num_sequences_total',
                           'sequences_file_size',
                           'sequences_num_records',
                           'heavy_cdr3_aa_max_length'):
            df[int_column] = df[int_column].astype(pd.Int64Dtype())

        return df

    # --- More file management

    def copy_to(self,
                dest_path: Path = Path.home() / 'oas-copy',
                include_paired: bool = True,
                include_unpaired: bool = True,
                include_subset_meta: bool = False,
                include_summaries: bool = False,
                include_sequences: bool = False,
                include_original_csv: bool = False,
                include_stats: bool = False,
                max_num_sequences: int = -1,
                unit_probability: float = 1,
                filtering_strategy: str = 'none',
                overwrite: bool = False,
                verbose: bool = False) -> List[dict]:

        # Avoid circular import
        from abbert2.oas.filtering import create_filters_from_name
        # Global filter states for the whole run
        filters = create_filters_from_name(filtering_strategy)

        def copy_subset(oas_subset: str):

            logs = []

            # --- Subset online collected metadata
            if include_subset_meta:
                for path in (self.oas_path / oas_subset).glob('bulk_download*'):
                    subset_path = dest_path / oas_subset
                    subset_path.mkdir(parents=True, exist_ok=True)
                    copy_but_do_not_overwrite(path, dest_path=subset_path, overwrite=overwrite)
            # --- Units
            for unit in self.units_in_disk(oas_subset=oas_subset):
                if 0 < unit_probability < 1:
                    seed = xxhash.xxh32_intdigest(unit.id_string)
                    if np.random.RandomState(seed=seed).uniform() > unit_probability:
                        continue  # unselected
                if unit.has_sequences:
                    print(f'COPYING {unit.id}')
                    logs += unit.copy_to(dest_path,
                                         include_sequences=include_sequences,
                                         include_original_csv=include_original_csv,
                                         include_stats=include_stats,
                                         max_num_sequences=max_num_sequences,
                                         filters=filters,
                                         overwrite=overwrite,
                                         verbose=verbose)

            # --- Summaries
            if include_summaries:
                summaries_path = dest_path / 'summaries'
                for path in (self.oas_path / 'summaries').glob('*'):
                    copy_but_do_not_overwrite(path, dest_path=summaries_path, overwrite=overwrite)

            print(f'Find your OAS dump in {dest_path}')

            return logs

        logs = []

        if include_paired:
            logs += copy_subset(oas_subset='paired')
        if include_unpaired:
            logs += copy_subset(oas_subset='unpaired')

        if verbose:
            print('=' * 80)
            print(f'Finished copying {self.oas_path} to {dest_path}')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                logs_df = pd.DataFrame(logs)
                for column in ('unfiltered_length', 'filtered_length', 'filtered_out'):
                    if column in logs_df:
                        logs_df[column] = logs_df[column].astype(pd.Int64Dtype())
                print(logs_df.round(2))
            print('=' * 80)

        return logs


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

        # A little complicated logic to ensure consistent definition of parent dataset
        # See also Aarti proposed changes allowing to take a path as input
        # (likely put it in a factory method)
        if oas_path is not None and oas is not None:
            if oas_path != oas.oas_path:
                raise Exception(f'OAS path inconsistency ({oas.oas_path} != {oas_path})')
        if oas_path is None:
            if oas is not None:
                oas_path = oas.oas_path
            else:
                oas_path = find_oas_path()
        self._oas_path = Path(oas_path)
        self._oas = oas

        # Precompute id-based unit hash
        self.unit_hash = xxhash.xxh3_64_intdigest(self.id_string)

    # --- Unit coordinates

    @property
    def id(self) -> Tuple[str, str, str]:
        return self.oas_subset, self.study_id, self.unit_id

    @property
    def id_string(self) -> str:
        return ','.join(self.id)

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

    def download(self,
                 force: bool = False,
                 dry_run: bool = True,
                 drop_caches: bool = True,
                 resume: bool = False):

        print(f'Downloading {self.id}')

        if not force and not self.needs_redownload:
            print(f'\tNo need to redownload {self.id}')
            return

        if drop_caches:
            print(f'\tRemove {self.id}: {self.path}')
            if not dry_run:
                try:
                    shutil.rmtree(self.path, ignore_errors=False)
                except FileNotFoundError:
                    pass

        if dry_run:
            print(f'\tDRY-RUN Not downloading {self.id}')
            return

        # Check that remote sizes coincide
        remote_size = self.online_csv_size_bytes
        remote_size_checked = int(
            requests.head(self.original_url, allow_redirects=True).headers.get('content-length', 0))
        if remote_size != remote_size_checked:
            raise Exception(f'Remote size must coincide with metadata size '
                            f'({remote_size_checked} != {remote_size})')

        # Configure resuming appropriately
        download_start_byte = 0
        if resume and self.has_original_csv:
            local_size = self.original_csv_path.stat().st_size
            if local_size < remote_size:
                download_start_byte = local_size

        with requests.get(self.original_url,
                          allow_redirects=True,
                          stream=True,
                          headers={'Range': f'bytes={download_start_byte}-'}) as request:
            self.path.mkdir(parents=True, exist_ok=True)
            open_mode = 'wb' if 0 == download_start_byte else 'ab'
            with self.original_csv_path.open(open_mode) as writer:
                with tqdm(total=remote_size,
                          unit='B',
                          unit_scale=True,
                          unit_divisor=1024,
                          desc=str(self.id),
                          initial=download_start_byte,
                          ascii=True, miniters=1) as pbar:
                    for chunk in request.iter_content(32 * 1024):
                        writer.write(chunk)
                        pbar.update(len(chunk))

    # --- Unit metadata

    @property
    def metadata_path(self) -> Path:
        return self.path / f'{self.unit_id}.metadata.json'

    def update_metadata(self):
        metadata = self.oas.unit_metadata(oas_subset=self.oas_subset,
                                          study_id=self.study_id,
                                          unit_id=self.unit_id)
        metadata = {k: to_json_friendly(v) for k, v in metadata.items()}
        self.persist_metadata(metadata)
        return metadata

    @cached_property
    def metadata(self):
        try:
            with self.metadata_path.open('rt') as reader:
                return json.load(reader)
        except (FileNotFoundError, IOError, JSONDecodeError):
            return self.update_metadata()

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
            'num_heavy_sequences', 'num_light_sequences', 'heavy_cdr3_aa_max_length'
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

    def sequences_df(self, columns=None, as_dataframe=True) -> Optional[Union[pd.DataFrame, Table]]:
        try:
            return from_parquet(self.sequences_path, columns=columns, as_dataframe=as_dataframe)
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
    def present_chains(self) -> Tuple[str, ...]:
        df = self.sequences_df(columns='chain')
        if df is None:
            return ()
        return tuple(sorted(df['chain'].unique()))

    def _schema_arrow(self):
        pq = self._pq()
        if pq is not None:
            return self._pq().schema_arrow
        return None

    def in_disk_column_names(self):
        schema = self._schema_arrow()
        if schema is not None:
            return schema.names
        return None

    @property
    def has_heavy_sequences(self) -> bool:
        return self.num_heavy_sequences > 0

    @property
    def has_light_sequences(self) -> bool:
        return self.num_light_sequences > 0

    @property
    def sequences_num_records(self) -> Optional[int]:
        if self.has_sequences:
            return self._pq().metadata.num_rows
        return None

    num_sequences = sequences_num_records

    def _num_chain_sequences(self, chain):
        df = self.sequences_df(columns='chain')
        if df is None:
            return 0
        return len(df.query(f'chain == "{chain}"'))

    @cached_property
    def num_heavy_sequences(self):
        return self._num_chain_sequences(chain='heavy')

    @cached_property
    def num_light_sequences(self):
        return self._num_chain_sequences(chain='light')

    @property
    def sequences_miss_processing(self) -> bool:
        num_records = self.sequences_num_records
        if num_records is None:
            return True
        # we could also check sum(redundancy) == theoretical_num_sequences_total...
        return num_records < self.theoretical_num_sequences_unique

    def region_max_length(self, region='cdr3', chain='heavy', aa=True) -> Optional[int]:
        pq = self._pq()
        if aa:
            region = region + '_aa'
        if pq is not None:
            if chain is None:
                # leverage parquet precomputed column stats
                column_index = pq.schema_arrow.get_field_index(f'{region}_length')
                if column_index != -1:
                    return max(pq.metadata.row_group(row_group).column(column_index).statistics.max
                               for row_group in range(pq.metadata.num_row_groups))
            df = self.sequences_df(columns=['chain', f'{region}_length']).query(f'chain == "{chain}"')
            if df is not None and len(df):
                return df[f'{region}_length'].max()
        return None

    @property
    def heavy_cdr1_max_length(self):
        return self.region_max_length(region='cdr1', chain='heavy', aa=False)

    @property
    def heavy_cdr1_aa_max_length(self):
        return self.region_max_length(region='cdr1', chain='heavy', aa=True)

    @property
    def heavy_cdr2_max_length(self):
        return self.region_max_length(region='cdr2', chain='heavy', aa=False)

    @property
    def heavy_cdr2_aa_max_length(self):
        return self.region_max_length(region='cdr2', chain='heavy', aa=True)

    @property
    def heavy_cdr3_max_length(self):
        return self.region_max_length(region='cdr3', chain='heavy', aa=False)

    @property
    def heavy_cdr3_aa_max_length(self):
        return self.region_max_length(region='cdr3', chain='heavy', aa=True)

    @property
    def light_cdr1_max_length(self):
        return self.region_max_length(region='cdr1', chain='light', aa=False)

    @property
    def light_cdr1_aa_max_length(self):
        return self.region_max_length(region='cdr1', chain='light', aa=True)

    @property
    def light_cdr2_max_length(self):
        return self.region_max_length(region='cdr2', chain='light', aa=False)

    @property
    def light_cdr2_aa_max_length(self):
        return self.region_max_length(region='cdr2', chain='light', aa=True)

    @property
    def light_cdr3_max_length(self):
        return self.region_max_length(region='cdr3', chain='light', aa=False)

    @property
    def light_cdr3_aa_max_length(self):
        return self.region_max_length(region='cdr3', chain='light', aa=True)

    # --- Consolidated stats

    @property
    def stats_path(self) -> Path:
        return self.path / f'{self.unit_id}.stats.pickle'

    def consolidated_stats(self, recompute: bool = False):

        cache_path = self.stats_path

        if not recompute:
            try:
                return pd.read_pickle(cache_path)
            except IOError:
                pass

        df = self.sequences_df()
        if df is None:
            return None

        stats = {}
        for chain, chain_df in df.groupby('chain'):
            aligned_position_counts = defaultdict(int)
            for sequence, positions, insertions in zip(chain_df['sequence_aa'],
                                                       chain_df['imgt_positions'],
                                                       chain_df['imgt_insertions']):
                if pd.isnull(sequence):
                    continue
                insertions = () if not isinstance(insertions, np.ndarray) else insertions
                # TODO this is a really tight loop we should move out python...
                for aa, position, insertion in zip_longest(sequence,
                                                           positions,
                                                           insertions,
                                                           fillvalue=''):
                    aligned_position_counts[f'{position}{insertion}={aa}'] += 1
            stats[chain] = {
                'aligned_position_counts': dict(aligned_position_counts),
                'sequence_aa_length_counts': df[f'sequence_aa'].str.len().value_counts().to_dict(),
            }
            for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4'):
                stats[chain][f'{region}_aa_length_counts'] = df[f'{region}_aa_length'].value_counts().to_dict()
            # TODO: collect other stats for things like QA, germlines...
        pd.to_pickle(stats, cache_path)
        return stats

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
                include_original_csv: bool = False,
                include_stats: bool = False,
                max_num_sequences: int = -1,
                filters: Sequence[Callable] = None,
                overwrite: bool = False,
                verbose: bool = True) -> Sequence[dict]:

        from abbert2.oas.filtering import filter_df

        logs = []

        oas_path = Path(oas_path)
        if oas_path == self.oas_path:
            raise Exception('Copying a unit over itself is not supported')

        dest_path = oas_path / self.oas_subset / self.study_id / self.unit_id

        # copy metadata
        start = time.perf_counter()
        copy_but_do_not_overwrite(self.metadata_path, dest_path, overwrite=overwrite)
        logs.append({
            'name': 'CopyUnitMetadata',
            'taken_s': time.perf_counter() - start,
            'unit': self.unit_id
        })

        # copy processed sequences
        if include_sequences and self.has_sequences:
            if max_num_sequences < 0 and not filters:
                start = time.perf_counter()
                copy_but_do_not_overwrite(self.sequences_path, dest_path, overwrite=overwrite)
                logs.append({
                    'name': 'CopyUnitSequences',
                    'taken_s': time.perf_counter() - start,
                    'unit': self.unit_id
                })
            else:
                dest = dest_path / self.sequences_path.name
                if dest.is_file() and not overwrite:
                    raise Exception(f'Path already exists and will not overwrite ({dest})')
                start = time.perf_counter()
                df = self.sequences_df()
                logs.append({
                    'name': 'ReadSequencesFromDisk',
                    'unfiltered_length': len(df),
                    'taken_s': time.perf_counter() - start,
                    'unit': self.unit_id
                })
                if max_num_sequences >= 0:
                    start = time.perf_counter()
                    df = df.sample(n=min(max_num_sequences, len(df)), random_state=19)
                    logs.append({
                        'name': 'SubsampleSequences',
                        'unfiltered_length': logs[-1]['unfiltered_length'],
                        'filtered_length': len(df),
                        'taken_s': time.perf_counter() - start,
                        'unit': self.unit_id
                    })
                if filters:
                    start = time.perf_counter()
                    # noinspection PyTypeChecker
                    df, filtering_logs = filter_df(df,
                                                   unit=self,
                                                   filters=filters,
                                                   keep_df_history=False,
                                                   verbose=verbose)
                    logs += filtering_logs
                    logs.append({
                        'name': 'ApplyFilters',
                        'unfiltered_length': logs[-1]['unfiltered_length'],
                        'filtered_length': len(df),
                        'taken_s': time.perf_counter() - start,
                        'unit': self.unit_id
                    })
                start = time.perf_counter()
                compress_sequences_df(df=df, path=dest)
                logs.append({
                    'name': 'WriteSequencesToDisk',
                    'unfiltered_length': logs[-1]['unfiltered_length'],
                    'taken_s': time.perf_counter() - start,
                    'unit': self.unit_id
                })
        if include_sequences:
            start = time.perf_counter()
            copy_but_do_not_overwrite(self.processing_logs_file, dest_path, overwrite=overwrite)
            copy_but_do_not_overwrite(self.processing_error_logs_file, dest_path, overwrite=overwrite)
            logs.append({
                'name': 'CopyLogs',
                'taken_s': time.perf_counter() - start
            })

        # copy original csv
        if include_original_csv:
            start = time.perf_counter()
            # +2: unit metadata and column names
            copy_but_do_not_overwrite(self.original_csv_path,
                                      dest_path,
                                      num_rows_header=max_num_sequences + 2,
                                      overwrite=overwrite)
            logs.append({
                'name': 'CopyOriginalCSV',
                'taken_s': time.perf_counter() - start
            })

        # copy processed stats (N.B., without recomputing for subsets)
        if include_stats:
            start = time.perf_counter()
            copy_but_do_not_overwrite(self.stats_path, dest_path, overwrite=overwrite)
            logs.append({
                'name': 'CopyStats',
                'taken_s': time.perf_counter() - start
            })

        return logs


# --- Entry points

def _consolidate_unit_stats(unit: Unit, recompute=False):
    if unit.has_sequences:
        print(f'Consolidating stats for {unit.id}')
        unit.consolidated_stats(recompute=recompute)


def consolidate_all_units_stats(oas_path: Optional[Union[str, Path]] = None,
                                recompute: bool = False,
                                n_jobs=-1):
    oas = OAS(oas_path=oas_path)
    Parallel(n_jobs=n_jobs)(
        delayed(_consolidate_unit_stats)(
            unit=unit,
            recompute=recompute,
        )
        for unit in oas.units_in_disk()
    )


def aligned_positions_to_df(aps):
    records = []
    for position_aa, count in aps.items():
        position, insertion, aa = parse_anarci_position_aa_to_imgt_code(position_aa)
        records.append({
            'position': str(position) + anarci_imgt_code_to_insertion(insertion),
            'aa': aa,
            'count': count
        })
    df = pd.DataFrame(records)  # .pivot(index='position', columns='aa')['count']
    df = df.groupby(['position', 'aa'])['count'].sum().unstack()

    return df


def summarize_count_stats(oas_path: Optional[Union[str, Path]] = None, recompute=False):
    oas = OAS(oas_path=oas_path)
    cache_path = oas.oas_path / 'summaries' / 'count_stats.pickle'

    if not recompute:
        try:
            return pd.read_pickle(cache_path)
        except IOError:
            ...

    # Initialize stats
    summarized_stats = {}
    for chain in ('heavy', 'light'):
        summarized_stats[chain] = {
            'aligned_position_counts': None,
            'sequence_aa_length_counts': None,
        }
        for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4'):
            summarized_stats[chain][f'{region}_aa_length_counts'] = None

    # Aggregate stats
    for unit in oas.units_in_disk():
        print(f'Aggregating counts for unit: {unit.id}')
        stats = unit.consolidated_stats()
        if stats is not None:
            for chain, chain_stats in stats.items():
                # --- Aligned position amino acid count distribution
                aligned_position_counts = aligned_positions_to_df(chain_stats['aligned_position_counts'])
                if summarized_stats[chain]['aligned_position_counts'] is None:
                    summarized_stats[chain]['aligned_position_counts'] = aligned_position_counts
                else:
                    summarized_stats[chain]['aligned_position_counts'] = (
                        aligned_position_counts.add(summarized_stats[chain]['aligned_position_counts'], fill_value=0)
                    )

                # --- Length histograms: full sequences
                sequence_length_counts = pd.Series(chain_stats['sequence_aa_length_counts'])
                if summarized_stats[chain]['sequence_aa_length_counts'] is None:
                    # noinspection PyTypeChecker
                    summarized_stats[chain]['sequence_aa_length_counts'] = sequence_length_counts
                else:
                    summarized_stats[chain]['sequence_aa_length_counts'] = (
                        sequence_length_counts.add(summarized_stats[chain]['sequence_aa_length_counts'], fill_value=0)
                    )
                # --- Length histograms: regions
                for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4'):
                    region_length_counts = pd.Series(chain_stats[f'{region}_aa_length_counts'])
                    if summarized_stats[chain][f'{region}_aa_length_counts'] is None:
                        # noinspection PyTypeChecker
                        summarized_stats[chain][f'{region}_aa_length_counts'] = region_length_counts
                    else:
                        summarized_stats[chain][f'{region}_aa_length_counts'] = (
                            region_length_counts.add(
                                summarized_stats[chain][f'{region}_aa_length_counts'], fill_value=0
                            )
                        )

    # Nice to have:
    #   - aligned aminoacid counts in the right order
    #   - a small indicator of the region (fwr1...)
    for chain in ('heavy', 'light'):
        # The chain aggregated stats
        chain_stats = summarized_stats[chain]
        # Amino acids alpha-sorted
        apc_df: Optional[pd.DataFrame] = chain_stats['aligned_position_counts']
        if apc_df is not None:
            apc_df = apc_df.sort_index(axis='columns')
            # Positions numbered-sorted
            apc_df = apc_df.loc[sorted(apc_df.index, key=parse_anarci_position_to_imgt_code)]
            # Use int type
            apc_df = apc_df.astype(pd.Int64Dtype())
            # Add a small indicator of the region
            from abnumber.common import SCHEME_POSITION_TO_REGION
            apc_df['region'] = apc_df.index.map(
                lambda x: SCHEME_POSITION_TO_REGION['imgt'][parse_anarci_position_to_imgt_code(x)[0]].lower()
            )
            apc_df = apc_df[['region'] + [column for column in apc_df.columns if column != 'region']]
            # Generate marginals
            apc_df['total'] = apc_df.sum(numeric_only=True, axis='columns').astype(pd.Int64Dtype())
            apc_df.loc['total'] = apc_df.sum(numeric_only=True, axis='index').astype(pd.Int64Dtype())
            apc_df.loc['total', 'region'] = 'sequence'
            # Done
            chain_stats['aligned_position_counts'] = apc_df
        # Histograms sorted in ascending order
        histograms = ['sequence_aa_length_counts']
        histograms += [f'{region}_aa_length_counts'
                       for region in ('fwr1', 'cdr1', 'fwr2', 'cdr2', 'fwr3', 'cdr3', 'fwr4')]
        for histogram in histograms:
            histogram_series: Optional[pd.Series] = chain_stats[histogram]
            if histogram_series is not None:
                chain_stats[histogram] = histogram_series.sort_index().astype(pd.Int64Dtype())
                # noinspection PyUnresolvedReferences
                chain_stats[histogram].index = chain_stats[histogram].index.map(int)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(summarized_stats, cache_path)

    return summarized_stats


def print_count_stats():
    counts = summarize_count_stats(recompute=False)
    for chain in ('heavy', 'light'):
        # noinspection PyTypeChecker
        df: pd.DataFrame = counts[chain]['aligned_position_counts']
        # What did we get?
        df.info()
        print(f'Number of amino acids ({chain}): {df.loc["total", "total"]}')
        print(f'Number of sequences   ({chain}): {df.loc["105", "total"]}')
        #
        # Number of amino acids (heavy): 151_036_497_653
        # Number of sequences   (heavy):   1_375_745_760
        # Number of amino acids (light):   3_956_316_690
        # Number of sequences   (light):      36_803_062
        #

        # noinspection PyTypeChecker
        counts[chain]['aligned_position_counts'] = df

        print(f'Number of positions: {df.shape[0]}')
        # 1240 different
        for region, region_df in df.groupby('region'):
            print(f'{region}:\tnum_positions={region_df.shape[0]:03d} num_tokens={region_df.shape[1]}')
        #
        # HEAVY:
        # cdr1:	num_positions=030 num_tokens=24
        # cdr2:	num_positions=034 num_tokens=24
        # cdr3:	num_positions=065 num_tokens=24
        # fwr1:	num_positions=186 num_tokens=24
        # fwr2:	num_positions=233 num_tokens=24
        # fwr3:	num_positions=623 num_tokens=24
        # fwr4:	num_positions=068 num_tokens=24
        # sequence:	num_positions=001 num_tokens=24
        #
        # LIGHT:
        # cdr1:	num_positions=028 num_tokens=23
        # cdr2:	num_positions=029 num_tokens=23
        # cdr3:	num_positions=026 num_tokens=23
        # fwr1:	num_positions=169 num_tokens=23
        # fwr2:	num_positions=219 num_tokens=23
        # fwr3:	num_positions=405 num_tokens=23
        # fwr4:	num_positions=027 num_tokens=23
        #

        # Build a logo plot and a stacked plot for different regions
        # Perhaps, collapse small tail positions
        # Do histogram, nice normalized plus marginals


# --- Maintenance


def compare_csv_schemas():
    """
    Compares the schemas of the original OAS CSVs.

    This function will print useful information and fail out loud with an exception if the expected schema changes.

    Only requires "cache_units_meta" and "populate_metadata" to have been run, not the CSVs in disk.

    Read the content in the function body to learn about expected schemas.
    """

    oas = OAS()

    # What columns did we find in the original CSVs?
    df = oas.unit_metadata_df
    df['column_names'] = df.column_names.apply(lambda x: tuple(sorted(set(x))))

    #
    # --- After running this on the 2023/02/04 OAS version, we get 4 groups of schemata
    #   - UNPAIRED_1: An unpaired schema with 15624 units
    #   - UNPAIRED_2: An unpaired schema with 7 units (study: Richardson_2022)
    #   - PAIRED_1: A paired schema with 47 units
    #   - PAIRED_2: A paired schema with 111 units (studies: Jaffe_2022, Mor_2021, Woodruff_2020)
    # What follows is a description of these groups
    #

    # 15624 unpaired units have these columns
    UNPAIRED_1_COLUMNS = (
        'ANARCI_numbering', 'ANARCI_status', 'Redundancy', 'c_region',
        'cdr1', 'cdr1_aa', 'cdr1_end', 'cdr1_start',
        'cdr2', 'cdr2_aa', 'cdr2_end', 'cdr2_start',
        'cdr3', 'cdr3_aa', 'cdr3_end', 'cdr3_start',
        'complete_vdj',
        'd_alignment_end', 'd_alignment_start', 'd_call', 'd_cigar',
        'd_germline_alignment', 'd_germline_alignment_aa', 'd_germline_end', 'd_germline_start',
        'd_identity', 'd_score', 'd_sequence_alignment', 'd_sequence_alignment_aa',
        'd_sequence_end', 'd_sequence_start', 'd_support',
        'fwr1', 'fwr1_aa', 'fwr1_end', 'fwr1_start',
        'fwr2', 'fwr2_aa', 'fwr2_end', 'fwr2_start',
        'fwr3', 'fwr3_aa', 'fwr3_end', 'fwr3_start',
        'fwr4', 'fwr4_aa', 'fwr4_end', 'fwr4_start',
        'germline_alignment', 'germline_alignment_aa',
        'j_alignment_end', 'j_alignment_start', 'j_call', 'j_cigar',
        'j_germline_alignment', 'j_germline_alignment_aa', 'j_germline_end', 'j_germline_start',
        'j_identity', 'j_score', 'j_sequence_alignment', 'j_sequence_alignment_aa',
        'j_sequence_end', 'j_sequence_start', 'j_support',
        'junction', 'junction_aa', 'junction_aa_length', 'junction_length',
        'locus',
        'np1', 'np1_length',
        'np2', 'np2_length',
        'productive', 'rev_comp',
        'sequence', 'sequence_alignment', 'sequence_alignment_aa',
        'stop_codon',
        'v_alignment_end', 'v_alignment_start', 'v_call', 'v_cigar',
        'v_frameshift',
        'v_germline_alignment', 'v_germline_alignment_aa', 'v_germline_end', 'v_germline_start',
        'v_identity', 'v_score', 'v_sequence_alignment', 'v_sequence_alignment_aa',
        'v_sequence_end', 'v_sequence_start', 'v_support',
        'vj_in_frame'
    )

    # UNPAIRED_1_STUDIES = (
    #     'Banerjee_2017', 'Bashford_2013', 'Bender_2020', 'Bernardes_2020',
    #     'Bernat_2019_1', 'Bernat_2019_2', 'Bhiman_2015', 'Bolland_2016', 'Bonsignori_2016',
    #     'Briney_2019', 'Buchheim_2020', 'Chen_2020', 'Collins_2015', 'Corcoran_2016',
    #     'Cui_2019', 'Davis_2019', 'Doria-Rose_2015', 'Eccles_2020', 'Eliyahu_2018',
    #     'Ellebedy_2016', 'Fisher_2017',
    #     'Galson_2015', 'Galson_2015a', 'Galson_2016', 'Galson_2016a', 'Galson_2020',
    #     'Ghraichy_2020', 'Gidoni_2019', 'Greiff_2014', 'Greiff_2015', 'Greiff_2017',
    #     'Gupta_2017', 'Halliley_2015', 'Huang_2016', 'Jaffe_2022', 'Jiang_2013',
    #     'Johnson_2018', 'Joyce_2016', 'Khan_2016', 'Kim_2020', 'King_2020_1', 'King_2020_2',
    #     'Kuri-Cervantes_2020', 'Levin_2016', 'Levin_2017', 'Li_2017', 'Liao_2013', 'Lindner_2015',
    #     'Meng_2017', 'Menzel_2014', 'Montague_2021', 'Mor_2021', 'Mroczek_2014', 'Mukhamedova_2021',
    #     'Nielsen_2020', 'Ohm-Laursen_2018', 'Ota_2010', 'Palanichamy_2014', 'Parameswaran_2014',
    #     'Prohaska_2018', 'Rettig_2018', 'Rubelt_2016', 'Schanz_2014', 'Schultheiss_2020',
    #     'Setliff_2018', 'Sevy_2019', 'Sheng_2017', 'Simonich_2020', 'Soto_2016', 'Soto_2019',
    #     'Stern_2014', 'Sundling_2014', 'Tipton_2015', 'Tong_2017', 'Turchaninova_2016',
    #     'Turner_2021', 'VanDuijn_2017', 'Vander_Heiden_2017', 'Vergani_2017',
    #     'Waltari_2018', 'Wesemann_2013', 'Woodruff_2020',
    #     'Wu_2011', 'Wu_2014', 'Wu_2015',
    #     'Zhou_2013', 'Zhou_2015', 'Zhu_2012', 'Zhu_2013'
    # )

    # 7 unpaired units have these columns
    UNPAIRED_2_COLUMNS = (
        'ANARCI_numbering', 'ANARCI_status', 'Redundancy', 'c_region',
        'cdr1', 'cdr1_aa', 'cdr1_end', 'cdr1_start',
        'cdr2', 'cdr2_aa', 'cdr2_end', 'cdr2_start',
        'cdr3', 'cdr3_aa', 'cdr3_end', 'cdr3_start',
        'd_alignment_end', 'd_alignment_start', 'd_call', 'd_cigar',
        'd_germline_alignment', 'd_germline_alignment_aa', 'd_germline_end', 'd_germline_start',
        'd_identity', 'd_score', 'd_sequence_alignment', 'd_sequence_alignment_aa',
        'd_sequence_end', 'd_sequence_start', 'd_support',
        'fwr1', 'fwr1_aa', 'fwr1_end', 'fwr1_start',
        'fwr2', 'fwr2_aa', 'fwr2_end', 'fwr2_start',
        'fwr3', 'fwr3_aa', 'fwr3_end', 'fwr3_start',
        'fwr4', 'fwr4_aa', 'fwr4_end', 'fwr4_start',
        'germline_alignment', 'germline_alignment_aa',
        'j_alignment_end', 'j_alignment_start', 'j_call', 'j_cigar',
        'j_germline_alignment', 'j_germline_alignment_aa', 'j_germline_end', 'j_germline_start',
        'j_identity', 'j_score', 'j_sequence_alignment', 'j_sequence_alignment_aa',
        'j_sequence_end', 'j_sequence_start',
        'j_support',
        'junction', 'junction_aa', 'junction_aa_length', 'junction_length',
        'locus',
        'np1', 'np1_length',
        'np2', 'np2_length',
        'productive', 'rev_comp',
        'sequence', 'sequence_alignment', 'sequence_alignment_aa',
        'stop_codon',
        'v_alignment_end', 'v_alignment_start', 'v_call', 'v_cigar',
        'v_germline_alignment', 'v_germline_alignment_aa', 'v_germline_end', 'v_germline_start',
        'v_identity', 'v_score', 'v_sequence_alignment', 'v_sequence_alignment_aa',
        'v_sequence_end', 'v_sequence_start', 'v_support', 'vj_in_frame'
    )
    # UNPAIRED_2_STUDIES = ('Richardson_2022',)

    # 47 paired units have these columns
    PAIRED_1_COLUMNS = (
        'ANARCI_numbering_heavy', 'ANARCI_numbering_light',
        'ANARCI_status_heavy', 'ANARCI_status_light',
        'cdr1_aa_heavy', 'cdr1_aa_light',
        'cdr1_end_heavy', 'cdr1_end_light',
        'cdr1_heavy', 'cdr1_light',
        'cdr1_start_heavy', 'cdr1_start_light',
        'cdr2_aa_heavy', 'cdr2_aa_light',
        'cdr2_end_heavy', 'cdr2_end_light',
        'cdr2_heavy', 'cdr2_light',
        'cdr2_start_heavy', 'cdr2_start_light',
        'cdr3_aa_heavy', 'cdr3_aa_light',
        'cdr3_end_heavy', 'cdr3_end_light',
        'cdr3_heavy', 'cdr3_light',
        'cdr3_start_heavy', 'cdr3_start_light',
        'd_alignment_end_heavy', 'd_alignment_end_light',
        'd_alignment_start_heavy', 'd_alignment_start_light',
        'd_call_heavy', 'd_call_light',
        'd_cigar_heavy', 'd_cigar_light',
        'd_germline_alignment_aa_heavy', 'd_germline_alignment_aa_light',
        'd_germline_alignment_heavy', 'd_germline_alignment_light',
        'd_germline_end_heavy', 'd_germline_end_light',
        'd_germline_start_heavy', 'd_germline_start_light',
        'd_identity_heavy', 'd_identity_light',
        'd_score_heavy', 'd_score_light',
        'd_sequence_alignment_aa_heavy', 'd_sequence_alignment_aa_light',
        'd_sequence_alignment_heavy', 'd_sequence_alignment_light',
        'd_sequence_end_heavy', 'd_sequence_end_light',
        'd_sequence_start_heavy', 'd_sequence_start_light',
        'd_support_heavy', 'd_support_light',
        'fwr1_aa_heavy', 'fwr1_aa_light', 'fwr1_end_heavy', 'fwr1_end_light',
        'fwr1_heavy', 'fwr1_light', 'fwr1_start_heavy', 'fwr1_start_light',
        'fwr2_aa_heavy', 'fwr2_aa_light', 'fwr2_end_heavy', 'fwr2_end_light',
        'fwr2_heavy', 'fwr2_light', 'fwr2_start_heavy', 'fwr2_start_light',
        'fwr3_aa_heavy', 'fwr3_aa_light', 'fwr3_end_heavy', 'fwr3_end_light',
        'fwr3_heavy', 'fwr3_light', 'fwr3_start_heavy', 'fwr3_start_light',
        'germline_alignment_aa_heavy', 'germline_alignment_aa_light',
        'germline_alignment_heavy', 'germline_alignment_light',
        'j_alignment_end_heavy', 'j_alignment_end_light',
        'j_alignment_start_heavy', 'j_alignment_start_light',
        'j_call_heavy', 'j_call_light', 'j_cigar_heavy', 'j_cigar_light',
        'j_germline_alignment_aa_heavy', 'j_germline_alignment_aa_light',
        'j_germline_alignment_heavy', 'j_germline_alignment_light',
        'j_germline_end_heavy', 'j_germline_end_light',
        'j_germline_start_heavy', 'j_germline_start_light',
        'j_identity_heavy', 'j_identity_light',
        'j_score_heavy', 'j_score_light',
        'j_sequence_alignment_aa_heavy', 'j_sequence_alignment_aa_light',
        'j_sequence_alignment_heavy', 'j_sequence_alignment_light',
        'j_sequence_end_heavy', 'j_sequence_end_light',
        'j_sequence_start_heavy', 'j_sequence_start_light',
        'j_support_heavy', 'j_support_light',
        'junction_aa_heavy', 'junction_aa_length_heavy',
        'junction_aa_length_light', 'junction_aa_light',
        'junction_heavy', 'junction_length_heavy',
        'junction_length_light', 'junction_light',
        'locus_heavy', 'locus_light',
        'np1_heavy', 'np1_length_heavy', 'np1_length_light', 'np1_light',
        'np2_heavy', 'np2_length_heavy', 'np2_length_light', 'np2_light',
        'productive_heavy', 'productive_light',
        'rev_comp_heavy', 'rev_comp_light',
        'sequence_alignment_aa_heavy', 'sequence_alignment_aa_light',
        'sequence_alignment_heavy', 'sequence_alignment_light',
        'sequence_heavy',
        'sequence_id_heavy', 'sequence_id_light',
        'sequence_light',
        'stop_codon_heavy', 'stop_codon_light',
        'v_alignment_end_heavy', 'v_alignment_end_light',
        'v_alignment_start_heavy', 'v_alignment_start_light',
        'v_call_heavy', 'v_call_light',
        'v_cigar_heavy', 'v_cigar_light',
        'v_germline_alignment_aa_heavy', 'v_germline_alignment_aa_light',
        'v_germline_alignment_heavy', 'v_germline_alignment_light',
        'v_germline_end_heavy', 'v_germline_end_light', 'v_germline_start_heavy', 'v_germline_start_light',
        'v_identity_heavy', 'v_identity_light', 'v_score_heavy', 'v_score_light',
        'v_sequence_alignment_aa_heavy', 'v_sequence_alignment_aa_light',
        'v_sequence_alignment_heavy', 'v_sequence_alignment_light',
        'v_sequence_end_heavy', 'v_sequence_end_light', 'v_sequence_start_heavy', 'v_sequence_start_light',
        'v_support_heavy', 'v_support_light',
        'vj_in_frame_heavy', 'vj_in_frame_light'
    )
    # PAIRED_1_STUDIES = ('Alsoiussi_2020', 'Eccles_2020', 'Goldstein_2019', 'King_2020_2', 'Setliff_2019')

    # 111 paired units have these columns
    PAIRED_2_COLUMNS = (
        'ANARCI_numbering_heavy', 'ANARCI_numbering_light',
        'ANARCI_status_heavy', 'ANARCI_status_light',
        'Isotype_heavy', 'Isotype_light',
        'Redundancy_heavy', 'Redundancy_light',
        'c_region_heavy', 'c_region_light',
        'cdr1_aa_heavy', 'cdr1_aa_light',
        'cdr1_end_heavy', 'cdr1_end_light',
        'cdr1_heavy', 'cdr1_light',
        'cdr1_start_heavy', 'cdr1_start_light',
        'cdr2_aa_heavy', 'cdr2_aa_light',
        'cdr2_end_heavy', 'cdr2_end_light',
        'cdr2_heavy', 'cdr2_light',
        'cdr2_start_heavy', 'cdr2_start_light',
        'cdr3_aa_heavy', 'cdr3_aa_light',
        'cdr3_end_heavy', 'cdr3_end_light',
        'cdr3_heavy', 'cdr3_light',
        'cdr3_start_heavy', 'cdr3_start_light',
        'complete_vdj_heavy', 'complete_vdj_light',
        'd_alignment_end_heavy', 'd_alignment_end_light',
        'd_alignment_start_heavy', 'd_alignment_start_light',
        'd_call_heavy', 'd_call_light',
        'd_cigar_heavy', 'd_cigar_light',
        'd_germline_alignment_aa_heavy', 'd_germline_alignment_aa_light',
        'd_germline_alignment_heavy', 'd_germline_alignment_light',
        'd_germline_end_heavy', 'd_germline_end_light',
        'd_germline_start_heavy', 'd_germline_start_light',
        'd_identity_heavy', 'd_identity_light',
        'd_score_heavy', 'd_score_light',
        'd_sequence_alignment_aa_heavy', 'd_sequence_alignment_aa_light',
        'd_sequence_alignment_heavy', 'd_sequence_alignment_light',
        'd_sequence_end_heavy', 'd_sequence_end_light',
        'd_sequence_start_heavy', 'd_sequence_start_light',
        'd_support_heavy', 'd_support_light',
        'fwr1_aa_heavy', 'fwr1_aa_light', 'fwr1_end_heavy', 'fwr1_end_light',
        'fwr1_heavy', 'fwr1_light', 'fwr1_start_heavy', 'fwr1_start_light',
        'fwr2_aa_heavy', 'fwr2_aa_light', 'fwr2_end_heavy', 'fwr2_end_light',
        'fwr2_heavy', 'fwr2_light', 'fwr2_start_heavy', 'fwr2_start_light',
        'fwr3_aa_heavy', 'fwr3_aa_light', 'fwr3_end_heavy', 'fwr3_end_light',
        'fwr3_heavy', 'fwr3_light', 'fwr3_start_heavy', 'fwr3_start_light',
        'fwr4_aa_heavy', 'fwr4_aa_light', 'fwr4_end_heavy', 'fwr4_end_light',
        'fwr4_heavy', 'fwr4_light', 'fwr4_start_heavy', 'fwr4_start_light',
        'germline_alignment_aa_heavy', 'germline_alignment_aa_light',
        'germline_alignment_heavy', 'germline_alignment_light',
        'j_alignment_end_heavy', 'j_alignment_end_light',
        'j_alignment_start_heavy', 'j_alignment_start_light',
        'j_call_heavy', 'j_call_light', 'j_cigar_heavy', 'j_cigar_light',
        'j_germline_alignment_aa_heavy', 'j_germline_alignment_aa_light',
        'j_germline_alignment_heavy', 'j_germline_alignment_light',
        'j_germline_end_heavy', 'j_germline_end_light',
        'j_germline_start_heavy', 'j_germline_start_light',
        'j_identity_heavy', 'j_identity_light',
        'j_score_heavy', 'j_score_light',
        'j_sequence_alignment_aa_heavy', 'j_sequence_alignment_aa_light',
        'j_sequence_alignment_heavy', 'j_sequence_alignment_light',
        'j_sequence_end_heavy', 'j_sequence_end_light',
        'j_sequence_start_heavy', 'j_sequence_start_light',
        'j_support_heavy', 'j_support_light',
        'junction_aa_heavy', 'junction_aa_length_heavy',
        'junction_aa_length_light', 'junction_aa_light',
        'junction_heavy', 'junction_length_heavy',
        'junction_length_light', 'junction_light',
        'locus_heavy', 'locus_light',
        'np1_heavy', 'np1_length_heavy', 'np1_length_light', 'np1_light',
        'np2_heavy', 'np2_length_heavy', 'np2_length_light', 'np2_light',
        'productive_heavy', 'productive_light',
        'rev_comp_heavy', 'rev_comp_light',
        'sequence_alignment_aa_heavy', 'sequence_alignment_aa_light',
        'sequence_alignment_heavy', 'sequence_alignment_light',
        'sequence_heavy',
        'sequence_id_heavy', 'sequence_id_light',
        'sequence_light',
        'stop_codon_heavy', 'stop_codon_light',
        'v_alignment_end_heavy', 'v_alignment_end_light',
        'v_alignment_start_heavy', 'v_alignment_start_light',
        'v_call_heavy', 'v_call_light',
        'v_cigar_heavy', 'v_cigar_light',
        'v_frameshift_heavy', 'v_frameshift_light',
        'v_germline_alignment_aa_heavy', 'v_germline_alignment_aa_light',
        'v_germline_alignment_heavy', 'v_germline_alignment_light',
        'v_germline_end_heavy', 'v_germline_end_light', 'v_germline_start_heavy', 'v_germline_start_light',
        'v_identity_heavy', 'v_identity_light', 'v_score_heavy', 'v_score_light',
        'v_sequence_alignment_aa_heavy', 'v_sequence_alignment_aa_light',
        'v_sequence_alignment_heavy', 'v_sequence_alignment_light',
        'v_sequence_end_heavy', 'v_sequence_end_light', 'v_sequence_start_heavy', 'v_sequence_start_light',
        'v_support_heavy', 'v_support_light',
        'vj_in_frame_heavy', 'vj_in_frame_light'
    )
    # PAIRED_2_STUDIES = ('Jaffe_2022', 'Mor_2021', 'Woodruff_2020')

    # --- Now let's compare the schemas

    EXPECTED_SCHEMAS = {
        'U1': UNPAIRED_1_COLUMNS,
        'U2': UNPAIRED_2_COLUMNS,
        'P1': PAIRED_1_COLUMNS,
        'P2': PAIRED_2_COLUMNS
    }

    print('PRESENT COLUMN GROUPS')
    print('---------------------')
    column_groups = df.groupby('column_names')
    for group, size in column_groups:
        print(column_groups.size()[group], tuple(sorted(column_groups['oas_subset'].unique()[group])))
        print(tuple(sorted(column_groups['study_id'].unique()[group])))
        print(group)
        print('-' * 80)
    print('=' * 80)

    print('PRESENT COLUMN GROUPS vs EXPECTED COLUMN GROUPS')
    print('-----------------------------------------------')
    expected_not_present: set = set(EXPECTED_SCHEMAS.values()) - set(column_groups.indices)
    present_not_expected: set = set(column_groups.indices) - set(EXPECTED_SCHEMAS.values())
    if expected_not_present or present_not_expected:
        for column_set in sorted(expected_not_present):
            print('MISSING EXPECTED COLUMN SET: ', column_set)
        for column_set in sorted(present_not_expected):
            print('UNEXPECTED COLUMN SET: ', column_set)
        raise ValueError('OAS Schema has diverged (see previous log)')
    print('=' * 80)

    print('COMPARISON BETWEEN PRESENT COLUMN GROUPS')
    print('----------------------------------------')
    for (name1, columns1), (name2, columns2) in combinations(EXPECTED_SCHEMAS.items(), 2):
        if name1.startswith('P') and not name2.startswith('P'):
            columns1 = set(x.replace('_heavy', '').replace('_light', '') for x in columns1)
        if name2.startswith('P') and not name1.startswith('P'):
            columns2 = set(x.replace('_heavy', '').replace('_light', '') for x in columns2)
        print(f'{name1} - {name2}: {sorted(set(columns1) - set(columns2))}')
        print(f'{name2} - {name1}: {sorted(set(columns2) - set(columns1))}')
    #
    # NOTE: WE CAN SIMPLIFY ANALYSIS BY LOOKING ONLY AT U1 and P2, WHICH ARE THE COLUMN SUPERSETS
    # (see U2 - U1 and P1 - P2 are empty)
    # -------------------------------------------------------------------------------------------
    #
    # U1 - U2: ['complete_vdj', 'v_frameshift']
    # U2 - U1: []
    #
    # U1 - P1: ['Redundancy', 'c_region', 'complete_vdj', 'fwr4', 'fwr4_aa', 'fwr4_end', 'fwr4_start', 'v_frameshift']
    # P1 - U1: ['sequence_id']
    #
    # U1 - P2: []
    # P2 - U1: ['Isotype', 'sequence_id']
    #
    # U2 - P1: ['Redundancy', 'c_region', 'fwr4', 'fwr4_aa', 'fwr4_end', 'fwr4_start']
    # P1 - U2: ['sequence_id']
    #
    # U2 - P2: []
    # P2 - U2: ['Isotype', 'complete_vdj', 'sequence_id', 'v_frameshift']
    #
    # P1 - P2: []
    # P2 - P1: ['Isotype_heavy', 'Isotype_light', 'Redundancy_heavy', 'Redundancy_light',
    #           'c_region_heavy', 'c_region_light', 'complete_vdj_heavy', 'complete_vdj_light',
    #           'fwr4_aa_heavy', 'fwr4_aa_light',
    #           'fwr4_end_heavy', 'fwr4_end_light',
    #           'fwr4_heavy', 'fwr4_light',
    #           'fwr4_start_heavy', 'fwr4_start_light',
    #           'v_frameshift_heavy', 'v_frameshift_light']
    #
    # Conclusions:
    #   - we need to compute ourselves redundancy in some of the paired sets
    #   - the lack of fwr4 in P1 is worrying
    #   - all the other missing columns, we probably can live with
    #     (Briney was fixed for fwr4, yay!, maybe it still is missing in the data)
    #   - lack of consistency is heartbreaking
    #


def compare_new_schemas():
    """Compares the schemas of the generated parquet files."""
    oas = OAS()

    column_sets = {}
    for unit in oas.units_in_disk():
        columns = unit.in_disk_column_names()
        if columns:
            unit_id = '-'.join(unit.id)
            column_sets.setdefault(tuple(sorted(columns)), []).append(unit_id)
    print(len(sorted(column_sets)))


def diagnose(recompute=False):
    """Shows some diagnose information, for example, what units have failed processing."""

    oas = OAS()
    oas.populate_metadata_jsons(recompute=recompute)
    df = oas.nice_unit_meta_df(recompute=recompute, normalize_species=True)

    units_to_redownload = sorted(unit.original_url for unit in df.query('needs_redownload').unit)
    print(f'{len(units_to_redownload)} units to redownload')
    print('\n'.join(units_to_redownload))
    print('-' * 80)

    units_missing_processing = sorted('/'.join(unit.id) for unit in df.query('sequences_miss_processing').unit)
    print(f'{len(units_missing_processing)} units missing processing')
    print('\n'.join(units_missing_processing))
    print('-' * 80)


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

#
# BLOSUM FOR ANTIBODIES? Gabi:
# Hiya! So - short answer: I don't think so.
#
# Long answer: The hypervariability of the CDRs is facilitated by these stretches containing
# sequence motifs on DNA level that are recognized by some enzyme which then introduced base
# changes that cause mutations. Because these exchanges have a certain probability to convert
# one given base into a specific other base, there must be a bias for certain amino acid exchanges.
# There's probably literature about these sequence motifs, but I to the best of my knowledge,
# that has never been made into an exchange matrix...
#
