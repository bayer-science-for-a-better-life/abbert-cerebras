from abbert2.oas.oas import populate_metadata_jsons, extract_processed_oas, consolidate_all_units_stats, \
    summarize_count_stats
from abbert2.oas.preprocessing import cache_oas_units_meta, process_units


def main():
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        cache_oas_units_meta,
        populate_metadata_jsons,
        process_units,
        extract_processed_oas,
        consolidate_all_units_stats,
        summarize_count_stats,
    ])
    parser.dispatch()


if __name__ == '__main__':
    main()
