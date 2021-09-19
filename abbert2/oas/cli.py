from abbert2.oas.preprocessing import cache_oas_units_meta, process_units


def main():
    import argh
    parser = argh.ArghParser()
    parser.add_commands([
        cache_oas_units_meta,
        process_units,
    ])
    parser.dispatch()

if __name__ == '__main__':
    main()
