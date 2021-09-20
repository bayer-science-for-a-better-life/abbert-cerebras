from pprint import pprint

from abbert2.oas import OAS, RELATIVE_OAS_TEST_DATA_PATH

# OAS is the entry point for the dataset
oas = OAS(oas_path=RELATIVE_OAS_TEST_DATA_PATH)

#
# It provides "units", which allow to access anything from metadata to logs
# and of course, the sequences...
#
for unit in oas.units_in_disk():

    #
    # Units are identified by a tuple (oas_subset, study_id, unit_id)
    #   - oas_subset is one of "paired" or "unpaired"
    #   - study_id is where the data was used; note some sequencing runs are present in several studies
    #   - unit_id usually identify a subject (human, mouse...) in a moment in time
    #
    print(f'UNIT: {unit.id}')

    #
    # Each unit comes with metadata that can be used to things like
    #   - Filtering or grouping the dataset
    #   - Inform the training process (e.g., via loss)?
    #
    pprint(unit.nice_metadata)

    # Note that unit provide convenient access to most of these data
    print(f'Unit disease={unit.disease}, vaccine={unit.vaccine}')

    #
    # Optionally, a unit will have already its sequences ready to be used
    #
    # Note that, in comparison with the previous version of the dataset,
    # now all the column of this dataframe are suffixed with one of
    # "_heavy" or "_light". A dataframe can contain both if it comes
    # from the "paired" subset.
    #
    if unit.has_sequences:
        df = unit.sequences_df()
        df.info()
        for chain in ('light', 'heavy'):
            try:
                print(f'An antibody {chain} chain: {df.iloc[0][f"aligned_sequence_{chain}"]}')
            except KeyError:
                ...

    print('-' * 80)
