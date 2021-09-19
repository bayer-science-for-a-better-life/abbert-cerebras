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
from builtins import IOError
from functools import cached_property
from itertools import chain
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, Union, Iterator

from abbert2.common import to_json_friendly
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

    # --- Factories

    def units(self, subset: str = None) -> Iterator['Unit']:
        if subset is None:
            yield from chain(self.units(subset='paired'), self.units(subset='unpaired'))
        else:
            check_oas_subset(subset)
            for study_path in sorted((self.oas_path / subset).glob('*')):
                if study_path.is_dir():
                    for unit_path in study_path.glob('*'):
                        if unit_path.is_dir():
                            yield Unit(oas_subset=subset,
                                       study_id=study_path.stem,
                                       unit_id=unit_path.stem,
                                       oas_path=self.oas_path,
                                       oas=self)


class Study:
    """Manage a single OAS study."""
    # No use for the time being
    ...


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

    @property
    def id(self) -> Tuple[str, str, str]:
        return self.oas_subset, self.study_id, self.unit_id

    @property
    def study_id(self) -> str:
        return self._study_id

    @property
    def unit_id(self) -> str:
        return self._unit_id

    @property
    def oas_subset(self) -> str:
        return self._oas_subset

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
    def _original_csv_path(self) -> Path:
        return self.path / f'{self.unit_id}.csv.gz'

    def has_original_csv(self) -> bool:
        return self._original_csv_path.is_file()

    # --- Unit metadata

    @property
    def _metadata_path(self) -> Path:
        return self.path / f'{self.unit_id}.metadata.json'

    @cached_property
    def metadata(self):
        try:
            with self._metadata_path.open('rt') as reader:
                return json.load(reader)
        except (FileNotFoundError, IOError, JSONDecodeError):
            metadata = self.oas.unit_metadata(oas_subset=self.oas_subset,
                                              study_id=self.study_id,
                                              unit_id=self.unit_id)
            metadata = {k: to_json_friendly(v) for k, v in metadata.items()}
            self.persist_metadata(metadata)
            return metadata

    def persist_metadata(self, metadata=None):
        if metadata is None:
            metadata = self.metadata  # Beware infinite recursion
        with self._metadata_path.open('wt') as writer:
            json.dump(metadata, writer, indent=2)

    # --- Sequences

    @property
    def _sequences_path(self):
        return self._metadata_path.with_suffix('.parquet')


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
    for unit in oas.units():
        print(unit.path)
        print(unit.has_original_csv())
        print(unit.metadata)
