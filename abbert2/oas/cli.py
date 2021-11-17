from abbert2.oas.oas import populate_metadata_jsons, extract_processed_oas, consolidate_all_units_stats, \
    summarize_count_stats, diagnose
from abbert2.oas.preprocessing import cache_units_meta, process_units, download_units, parse_all_anarci_status


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
        extract_processed_oas,
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
