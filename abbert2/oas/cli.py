from pathlib import Path
from typing import Optional, Union

from abbert2.oas.oas import consolidate_all_units_stats, summarize_count_stats, diagnose, OAS
from abbert2.oas.preprocessing import cache_units_meta, process_units, download_units, parse_all_anarci_status


def populate_metadata_jsons(oas_path: Optional[Union[str, Path]] = None):
    OAS(oas_path=oas_path).populate_metadata_jsons()


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
         overwrite: bool = False):
    OAS(oas_path=oas_path).copy_to(dest_path / f'filters={filtering_strategy}',
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
                                   overwrite=overwrite)


def main():
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        # --- Bootstrap commmands
        # Step 0: download bulk_download from the OAS website (not easy to automate)
        # Step 1: Run this to cache units metadata from the web
        cache_units_meta,
        # Step 2: Run this to populate individual unit metadata
        populate_metadata_jsons,
        # Step 3: Download the units (will take quite a while)
        download_units,
        # Step 4: Convert the CSVs to more efficient representations (will take a lot of time)
        process_units,

        # --- Maintenance commands
        # This allows to extract a copy of the processed units (e.g., to create tars)
        copy,
        # This prints reports on the units
        diagnose,
        # This attemps to parse all collected anarci status
        parse_all_anarci_status,

        # --- Analysis commands
        # This will consolidate stats for all units (e.g., frequences of amino acids on numbered sequences)
        consolidate_all_units_stats,
        # This will summarize and print count stats
        summarize_count_stats,
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()
