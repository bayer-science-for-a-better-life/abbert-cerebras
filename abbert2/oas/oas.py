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
from typing import Tuple, Union, Iterator, Optional, List

import pandas as pd
from pyarrow import ArrowInvalid
import pyarrow.parquet as pq

from abbert2.common import to_json_friendly, from_parquet, mtime, to_parquet
from abbert2.oas.common import find_oas_path, check_oas_subset


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
        return self._oas_path

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

    def units_in_meta(self) -> Iterator['Unit']:
        df = self.unit_metadata_df
        for oas_subset, study_id, unit_id in zip(df['oas_subset'], df['study_id'], df['unit_id']):
            yield self.unit(oas_subset=oas_subset, study_id=study_id, unit_id=unit_id)


class Study:
    """Manage a single OAS study."""
    # No use for the time being
    ...


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
            'sequencing_run', 'publication_link', 'study_author',
            'species', 'age',
            'bsource', 'btype',
            'subject', 'disease', 'vaccine',
            'longitudinal',
            'chain', 'isotype',
            'theoretical_num_sequences_unique', 'theoretical_num_sequences_total',
            'original_url', 'has_original_csv', 'original_local_csv_mdate', 'download_error', 'needs_redownload',
            'sequences_file_size', 'sequences_num_records', 'sequences_miss_processing',
            'heavy_cdr3_max_length'
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
    def species(self) -> Optional[str]:
        return self.metadata.get('Species')

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
        return self.sequences_path.is_file()

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
        if self.has_sequences:
            return pq.ParquetFile(self.sequences_path)
        return None

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
                if dest.is_file():
                    raise Exception(f'Path already exists and will not overwrite ({dest})')
                df = self.sequences_df()
                df = df.sample(n=min(max_num_sequences, len(df)), random_state=19)
                to_parquet(df, dest)
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

    for unit in oas.units_in_disk():
        print(unit.path)
        print(unit.metadata)
        print(unit.nice_metadata)
        if unit.has_sequences:
            unit.sequences_df().info()
            unit.copy_to(Path.home() / 'small-oas-deleteme', max_num_sequences=100, overwrite=True)
        assert unit == unit
