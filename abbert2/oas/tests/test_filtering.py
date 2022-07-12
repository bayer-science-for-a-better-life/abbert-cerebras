import pandas as pd

from abbert2.oas.common import RELATIVE_OAS_TEST_DATA_PATH

from abbert2.oas import OAS
from abbert2.oas.filtering import OnlyIsotypes


def test_only_isotypes():
    oas = OAS(RELATIVE_OAS_TEST_DATA_PATH)
    only_aghm_filter = OnlyIsotypes(isotypes=('IGHM',))
    for unit in oas.units_in_disk():
        original_df = unit.sequences_df()
        original_df_copy = original_df.copy()
        filtered_df, log = only_aghm_filter(original_df, unit)
        if unit.isotype != 'IGHM':
            assert not len(filtered_df)
            assert log['unfiltered_length'] == len(original_df)
            assert log['filtered_length'] == 0
            assert log['filtered_out'] == len(original_df)
        else:
            assert len(filtered_df) == len(original_df_copy)
            assert log['unfiltered_length'] == len(original_df)
            assert log['filtered_length'] == len(original_df)
            assert log['filtered_out'] == 0
        # No side effects
        pd.testing.assert_frame_equal(original_df, original_df_copy)
        # Good naming
        assert log['name'] == "OnlyIsotypes(isotypes=['IGHM'])"
