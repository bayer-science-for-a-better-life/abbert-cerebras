from pathlib import Path
from typing import Optional, Union

import pandas as pd

from abbert2.common import no_ssl_verification
from abbert2.oas.oas import consolidate_all_units_stats, summarize_count_stats, diagnose, OAS, compare_csv_schemas
from abbert2.oas.preprocessing import cache_units_meta, process_units, download_units, check_csv_parsing_corner_cases
from antidoto.misc import envvar


def populate_metadata(oas_path: Optional[Union[str, Path]] = None,
                      no_jsons: bool = False,
                      no_nice_unit_meta: bool = False,
                      no_normalize_species: bool = False,
                      recompute: bool = False):
    oas = OAS(oas_path=oas_path)
    if not no_jsons:
        oas.populate_metadata_jsons(recompute=recompute)
    if not no_nice_unit_meta:
        oas.nice_unit_meta_df(recompute=recompute, normalize_species=not no_normalize_species)


def copy(oas_path: Optional[Union[str, Path]] = None,
         dest_path: Path = Path.home() / 'oas-copy',
         no_paired: bool = False,
         no_unpaired: bool = False,
         include_subset_meta: bool = False,
         include_summaries: bool = False,
         include_sequences: bool = False,
         include_original_csv: bool = False,
         include_stats: bool = False,
         max_num_sequences: int = -1,
         unit_probability: float = 1.0,
         filtering_strategy: str = 'none',
         recompute: bool = False,
         verbose: bool = False):
    oas = OAS(oas_path=oas_path)
    dest_path = Path(dest_path) / f'filters={filtering_strategy}'
    logs = oas.copy_to(dest_path=dest_path,
                       include_paired=not no_paired,
                       include_unpaired=not no_unpaired,
                       include_subset_meta=include_subset_meta,
                       include_summaries=include_summaries,
                       include_sequences=include_sequences,
                       include_original_csv=include_original_csv,
                       include_stats=include_stats,
                       max_num_sequences=max_num_sequences,
                       unit_probability=unit_probability,
                       filtering_strategy=filtering_strategy,
                       overwrite=recompute,
                       verbose=verbose)
    pd.to_pickle(logs, Path(dest_path) / 'copy-logs.pickle')


def main():
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        # --- Bootstrap commmands

        # Step 0: download bulk_download from the OAS website (not easy to automate)

        # Step 1: Run this to cache units metadata from the web
        cache_units_meta,

        # Step 2: Run this to populate individual unit metadata
        populate_metadata,
        # Step 2b: Run compare_csv_schemas (see below) to detect schema changes

        # Step 3: Download the units (will take quite a while)
        download_units,

        # Step 4: Convert the CSVs to more efficient representations (will take a lot of time)
        # Run check_csv_parsing_corner_cases (see below) for some lean smoke testing
        process_units,
        # Step 4b: Run parse_all_anarci_status (see below) to detect ANARCI status parsing problems

        # Step 5: Run this again, recomputing, to populate unit metadata with processed sequences info
        # populate_metadata,

        # --- Maintenance commands
        # This will show what columns are in different units and screams if there is a change on expected schema
        compare_csv_schemas,
        # This command runs some smoke tests against exemplary units and screams if something is wrong
        check_csv_parsing_corner_cases,
        # This allows to extract a copy of the processed units (e.g., to create tars)
        copy,
        # This prints reports on the units
        diagnose,
        # This attemps to parse all collected anarci status
        # parse_all_anarci_status,

        # --- Analysis commands
        # This will consolidate stats for all units (e.g., frequences of amino acids on numbered sequences)
        consolidate_all_units_stats,
        # This will summarize and print count stats
        summarize_count_stats,
    ])
    parser.dispatch()


# This is an example workflow while developing / rerunning stuff
def example_main():
    with no_ssl_verification():
        with envvar('OAS_PATH', value=str(Path.home() / 'oas-new' / '20230204')):
            cache_units_meta(paired=False, n_jobs=1)
            populate_metadata()
            compare_csv_schemas()
            download_units(clean_not_in_meta=True)
            check_csv_parsing_corner_cases()
            process_units()
            populate_metadata(recompute=True)  # N.B. recompute now takes time, do once


if __name__ == '__main__':
    main()
