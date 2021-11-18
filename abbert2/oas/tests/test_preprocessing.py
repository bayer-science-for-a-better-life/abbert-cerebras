import numpy as np
import pytest

from abbert2.oas.preprocessing import parse_anarci_status

ANARCI_STATUS_EXPECTATIONS = [
    (
        '|Deletions: 1, 2||Missing Conserved Cysteine: 23|Shorter than IMGT defined: fw1|',
        {
            'deletions': np.array([1, 2], dtype=np.uint8),
            'missing_conserved_cysteine':  True,
            'fw1_shorter_than_imgt_defined': True
        }
    ),

    (
        '|Deletions: 1, 2|||Shorter than IMGT defined: fw1|CDR3 is over 37 aa long',
        {
            'deletions': np.array([1, 2], dtype=np.uint8),
            'fw1_shorter_than_imgt_defined': True,
            'cdr3_is_over_37_aa_long': True,
        }
    ),

    (
        "['Missing Conserved Cysteine 23 or 104', 'fw1 is shorter than IMGT defined']",
        {
            'missing_conserved_cysteine': True,
            'fw1_shorter_than_imgt_defined': True
        }
    ),

    (
        "['Unusual amino acid: X27', 'Unusual amino acid: X38', 'fw1 is shorter than IMGT defined']",
        {
            'unusual_residue': True,
            'fw1_shorter_than_imgt_defined': True
        }
    ),

    (
        "['fw1 is shorter than IMGT defined', 'fw4 is shorter than IMGT defined']",
        {
            'fw1_shorter_than_imgt_defined': True,
            'fw4_shorter_than_imgt_defined': True
        }
    ),

    (
        "['Unusual amino acid: X57', 'fw1 is shorter than IMGT defined']",
        {
            'unusual_residue': True,
            'fw1_shorter_than_imgt_defined': True,
        }
    ),

    (None, {}),
]


@pytest.mark.parametrize('status,expected', ANARCI_STATUS_EXPECTATIONS)
def test_parse_anarci_status(status, expected):
    np.testing.assert_equal(expected, parse_anarci_status(status))
