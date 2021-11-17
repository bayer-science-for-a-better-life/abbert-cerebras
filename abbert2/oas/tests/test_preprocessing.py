import numpy as np
import pytest

from abbert2.oas.preprocessing import parse_anarci_status

ANARCI_STATUS_EXPECTATIONS = [
    (
        '|Deletions: 1, 2||Missing Conserved Cysteine: 23|Shorter than IMGT defined: fw1|',
        {
            'deletions': np.array([1, 2], dtype=np.uint8),
            'missing_conserved_cysteine':  np.array([23], dtype=np.uint8),
            'shorter_than_imgt_defined': 'fw1'
        }
    ),

    (None, {}),
]


@pytest.mark.parametrize('status,expected', ANARCI_STATUS_EXPECTATIONS)
def test_parse_anarci_status(status, expected):
    np.testing.assert_equal(expected, parse_anarci_status(status))
