"""
Processing the original OAS data files into more convenient formats.
"""
import datetime
import json
import queue
import threading
import time
import traceback
from ast import literal_eval
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Union, Tuple, Dict, Optional, List
from urllib.parse import urlparse, unquote

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from more_itertools import distribute
from requests import HTTPError
from smart_open import open

from abbert2.common import to_parquet, from_parquet, parse_anarci_position, anarci_insertion_to_code
from abbert2.oas.common import find_oas_path, compress_sequences_df
from abbert2.oas.oas import OAS, Unit


#
# --- Manipulation of OAS "bulk_download.sh" and unit metadata
#
# To get "bulk_download.sh" for *unpaired* data:
#   - Go to http://opig.stats.ox.ac.uk/webapps/oas/oas
#   - Click search without adding any filter
#   - Download the shell script and put it under "<oas_root>/unpaired"
#
# To get "bulk_download.sh" for *paired* data:
#   - Go to http://opig.stats.ox.ac.uk/webapps/oas/oas_paired
#   - Click search without adding any filter
#   - Download the shell script and put it under "<oas_root>/paired"
#

def _parse_oas_url(url: str) -> Dict[str, str]:
    """
    Parses a OAS URL as present in `bulk_download.sh`.

    This function has also heuristics to detect changes in the URLs
    that might be indicative of changes to the dataset.

    Returns
    -------
    A dictionary with the following keys:
      - oas_subset is either "paired" or "unpaired"
      - study_id is the OAS study ID (typically name of first author_year of publication)
      - unit_id is the OAS unit ID, currently defined by the filename without extension
      - file_name is the file name of the unit
      - origin_id is the ID to the raw data (usually a SRA id (https://www.ncbi.nlm.nih.gov/sra))
      - origin_occurrence is an optional specialiser for the origin (like "1")
      - chain: "Heavy" or "Light" for unpaired, None for paired
      - isotype: "Bulk" for "Light", one of ("IGHA", "IGHG", "IGHM"...) if "Heavy", None if unpaired

    Examples
    --------
    >>> url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Setliff_2019/csv/SRR10313332_paired.csv.gz'
    >>> expected = {'oas_subset': 'paired',
    ...             'study_id': 'Setliff_2019',
    ...             'unit_id': 'SRR10313332_paired',
    ...             'file_name': 'SRR10313332_paired.csv.gz',
    ...             'origin_id': 'SRR10313332',
    ...             'origin_occurrence': None,
    ...             'chain': None,
    ...             'isotype': None}
    >>> _parse_oas_url(url) == expected
    True

    >>> url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz'
    >>> expected = {'oas_subset': 'unpaired',
    ...             'study_id': 'Eliyahu_2018',
    ...             'unit_id': 'ERR2843400_Heavy_IGHE',
    ...             'file_name': 'ERR2843400_Heavy_IGHE.csv.gz',
    ...             'origin_id': 'ERR2843400',
    ...             'origin_occurrence': None,
    ...             'chain': 'Heavy',
    ...             'isotype': 'IGHE'}
    >>> _parse_oas_url(url) == expected
    True

    """
    result = urlparse(_fix_oas_paired_url(url))
    if result.scheme != 'http':
        raise ValueError(f'URL {url} scheme must be http ({url})')
    if result.netloc != 'opig.stats.ox.ac.uk':
        raise ValueError(f'OAS seems to have been moved from "opig.stats.ox.ac.uk", update check ({url})')

    parts = PurePosixPath(unquote(result.path)).parts
    if 7 != len(parts):
        raise ValueError(f'Expected 7 parts in the path for {url}, got ({len(parts)})')
    if parts[:3] != ('/', 'webapps', 'ngsdb') or parts[5] != 'csv':
        raise ValueError(f'Expected fixed parts do not match {url}')

    *_, oas_subset, study_id, data_format, file_name = parts

    if oas_subset not in ('unpaired', 'paired'):
        raise ValueError(f'OAS subset not in ("paired", "unpaired") ({url})')

    if oas_subset == 'paired':
        origin_id, origin_occurrence, chain, isotype = file_name.split('_')[0], None, None, None
    else:
        parts = file_name.partition('.')[0].split('_')
        try:
            # Like: ERR2843400_Heavy_IGHE.csv.gz
            origin_id, chain, isotype = parts
            origin_occurrence = None
        except ValueError:
            try:
                # Like: ERR1759659_1_Heavy_Bulk.csv.gz
                # TODO: ask how to interpret these (1)
                origin_id, origin_occurrence, chain, isotype = parts
            except ValueError:
                # Like: rettig_2018_04_Heavy_Bulk.csv.gz
                # TODO: ask how to interpret these (2018, 04)
                study_id_one, study_id_two, origin_occurrence, chain, isotype = parts
                origin_id = study_id_one + study_id_two

    return {'oas_subset': oas_subset,
            'study_id': study_id,
            'unit_id': file_name.replace('.csv.gz', ''),
            'file_name': file_name,
            'origin_id': origin_id,
            'origin_occurrence': origin_occurrence,
            'chain': chain,
            'isotype': isotype}


def _fix_oas_paired_url(url: str) -> str:
    """
    Fixes OAS paired URLs.

    On 2021/09/10 there is a bug in the generation of bulk_download.sh for paired units.
    This function should fix it, and should be applicable to any OAS URL.

    Examples
    --------
    >>> broken_paired_url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/pairedSetliff_2019/csv/SRR10313332_paired.csv.gz'
    >>> _fix_oas_paired_url(broken_paired_url)
    'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Setliff_2019/csv/SRR10313332_paired.csv.gz'

    >>> fixed_paired_url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Setliff_2019/csv/SRR10313332_paired.csv.gz'
    >>> _fix_oas_paired_url(fixed_paired_url)
    'http://opig.stats.ox.ac.uk/webapps/ngsdb/paired/Setliff_2019/csv/SRR10313332_paired.csv.gz'

    >>> unpaired_url = 'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz'
    >>> _fix_oas_paired_url(unpaired_url)
    'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Eliyahu_2018/csv/ERR2843400_Heavy_IGHE.csv.gz'
    """
    if 'ngsdb/paired/' not in url:
        return url.replace('ngsdb/paired', 'ngsdb/paired/')
    return url


def _fix_bulk_download(path) -> List[str]:
    """
    Returns a list of wget commands, one per OAS unit to download.

    Takes care of "fixing" the download scripts, returning a deduplicated set of valid urls.

    As of 2021/09/10, there are two "problems" with these scripts:
      - Broken URLs for the paired subset (simply forgotten to add “/” after “paired”)
      - Duplicated file names in the unpaired subset (same SRA sequencing depositions in different studies)
    """
    path = Path(path)
    urls = []
    with path.open('rt') as reader:
        for line in reader:
            line = line.strip()
            if line:
                # Extract line
                _, url = line.split()
                # Sanity check
                _parse_oas_url(url)
                # Add
                urls.append(_fix_oas_paired_url(url))

    # Detect duplicates
    deduplicated_urls = sorted(set(urls))
    assert len(deduplicated_urls) == len(urls)
    #
    # But note there are duplicated file names in the unpaired subset.
    # These seem to correspond to the same SRA deposition used in different studies.
    # An example:
    #
    # unpaired ❯❯❯ ls -l ERR1812282*
    #   535643 Jul 29 00:42 ERR1812282_Heavy_Bulk.csv.gz
    #   547554 Aug  7 12:24 ERR1812282_Heavy_Bulk.csv.gz.1
    # 53716121 Jul 28 23:49 ERR1812282_Heavy_IGHA.csv.gz
    # 53804472 Aug  7 11:32 ERR1812282_Heavy_IGHA.csv.gz.1
    # 24014928 Jul 28 22:20 ERR1812282_Heavy_IGHE.csv.gz
    # 24019976 Aug  7 10:03 ERR1812282_Heavy_IGHE.csv.gz.1
    # 97540778 Jul 28 21:21 ERR1812282_Heavy_IGHG.csv.gz
    # 97668180 Aug  7 09:04 ERR1812282_Heavy_IGHG.csv.gz.1
    # 70807271 Jul 28 21:41 ERR1812282_Heavy_IGHM.csv.gz
    # 70747184 Aug  7 09:25 ERR1812282_Heavy_IGHM.csv.gz.1
    #
    # Corresponding to:
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Levin_2017/csv/ERR1812282_Heavy_Bulk.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Levin_2017/csv/ERR1812282_Heavy_IGHM.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Levin_2017/csv/ERR1812282_Heavy_IGHE.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Levin_2017/csv/ERR1812282_Heavy_IGHA.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Levin_2017/csv/ERR1812282_Heavy_IGHG.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Thornqvist_2018/csv/ERR1812282_Heavy_Bulk.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Thornqvist_2018/csv/ERR1812282_Heavy_IGHM.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Thornqvist_2018/csv/ERR1812282_Heavy_IGHE.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Thornqvist_2018/csv/ERR1812282_Heavy_IGHA.csv.gz
    # wget http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Thornqvist_2018/csv/ERR1812282_Heavy_IGHG.csv.gz
    #
    # So the issue is having the same RSA deposition across studies.
    # An interesting question is if these get the same antibody information in OAS.
    #

    return deduplicated_urls


def _download_units_info(urls: List[str],
                         job_id: int = -1,
                         verbose: bool = True,
                         add_study_metadata: bool = True) -> pd.DataFrame:
    import requests
    with requests.Session() as session:
        headers = []
        url: str
        for i, url in enumerate(urls):
            header = {'url': url}
            header.update(_parse_oas_url(url))
            header.update(session.head(url).headers)
            if add_study_metadata:
                unit_metadata = _read_unit_metadata(url, add_column_names=True)
                assert not (set(unit_metadata) & set(header))
                header.update(unit_metadata)
            if verbose:
                print(f'job={job_id} unit={i + 1}/{len(urls)}: {header}')
            headers.append(header)
        return pd.DataFrame(headers)


def oas_units_meta(oas_path: Union[str, Path] = None,
                   paired: bool = None,
                   keep_missing: bool = False,
                   recompute: bool = False,
                   n_jobs: int = -1) -> pd.DataFrame:
    """
    Returns a pandas dataframe with the metadata collected online from the units CSVs.

    The dataframe looks like:
    ---
    RangeIndex: 12694 entries, 0 to 12693
    Data columns (total 29 columns):
     #   Column             Non-Null Count  Dtype
    ---  ------             --------------  -----
     0   oas_subset         12694 non-null  object
     1   study_id           12694 non-null  object
     2   unit_id            12694 non-null  object
     3   url                12694 non-null  object
     4   file_name          12694 non-null  object
     5   origin_id          12694 non-null  object
     6   origin_occurrence  902 non-null    object
     7   chain              12647 non-null  object
     8   Date               12694 non-null  object
     9   Last-Modified      12694 non-null  object
     10  Content-Length     12694 non-null  object
     11  Run                12694 non-null  object
     12  Link               12694 non-null  object
     13  Author             12694 non-null  object
     14  Species            12694 non-null  object
     15  BSource            12694 non-null  object
     16  BType              12694 non-null  object
     17  Longitudinal       12694 non-null  object
     18  Age                12694 non-null  object
     19  Disease            12694 non-null  object
     20  Subject            12694 non-null  object
     21  Vaccine            12694 non-null  object
     22  Chain              12694 non-null  object
     23  Unique sequences   12694 non-null  Int64
     24  Isotype            12694 non-null  object
     25  Total sequences    12647 non-null  Int64
     26  Organism           157 non-null    object
     27  column_names       12694 non-null  object
     28  http_error         0 non-null      object
    dtypes: Int64(2), object(27)
    memory usage: 2.8+ MB
    {'Age': 'no',
     'Author': 'Banerjee et al., 2017',
     'BSource': 'Spleen',
     'BType': 'Unsorted-B-Cells',
     'Chain': 'Heavy',
     'Content-Length': '680524887',
     'Date': 'Sun, 14 Nov 2021 09:31:20 GMT',
     'Disease': 'None',
     'Isotype': 'Bulk',
     'Last-Modified': 'Thu, 29 Jul 2021 11:26:52 GMT',
     'Link': 'https://doi.org/10.1016/j.virol.2017.02.015',
     'Longitudinal': 'Terminal-bleed',
     'Organism': None,
     'Run': 'SRR5060321',
     'Species': 'rabbit',
     'Subject': 'no',
     'Total sequences': 2655033,
     'Unique sequences': 1671672,
     'Vaccine': 'HIV',
     'chain': 'Heavy',
     'column_names': ['sequence',
                      ...                # many columns...
                      'ANARCI_status'],
     'file_name': 'SRR5060321_Heavy_Bulk.csv.gz',
     'http_error': None,
     'oas_subset': 'unpaired',
     'origin_id': 'SRR5060321',
     'origin_occurrence': None,
     'study_id': 'Banerjee_2017',
     'unit_id': 'SRR5060321_Heavy_Bulk',
     'url': 'http://opig.stats.ox.ac.uk/webapps/ngsdb/unpaired/Banerjee_2017/csv/SRR5060321_Heavy_Bulk.csv.gz'}
    ---

    Parameters
    ----------
    oas_path : string or Path, default None
      The path to the local OAS copy

    paired : bool, default None
      If True, return only the paired subset
      If False, return only the unpaired subset
      If None, return both subsets

    keep_missing : bool, default False
      If True, units with missing online presence (i.e., non existent URLs) are kept

    recompute : bool, default False
      If True redownload the metadata from the online CSVs
      Otherwise try to use cached data

    n_jobs : int, default -1
      Number of jobs to use when collecting metadata online (joblib semantics)
    """

    if paired is None:
        # merge both paired and unpaired subsets
        subset_dfs = []
        for paired in (False, True):
            subset_dfs.append(
                oas_units_meta(oas_path=oas_path,
                               recompute=recompute,
                               paired=paired,
                               keep_missing=keep_missing,
                               n_jobs=n_jobs)
            )
        return pd.concat(subset_dfs).reset_index(drop=True)

    if oas_path is None:
        oas_path = find_oas_path()
    path = Path(oas_path) / ('paired' if paired else 'unpaired')

    # check the download script exists
    bulk_download_path = path / 'bulk_download.sh'
    if not bulk_download_path.is_file():
        help_text = (f'To get the download script for {"*unpaired*" if not paired else "*paired*"} data:\n'
                     f'  - Go to http://opig.stats.ox.ac.uk/webapps/oas/{"oas" if not paired else "oas_paired"}\n'
                     f'  - Click search without adding any filter\n'
                     f'  - Download the shell script and copy it to "{path.absolute()}"\n')
        print(help_text)
        raise Exception(f'{path} does not exist')

    # parse it
    urls = _fix_bulk_download(bulk_download_path)

    # read cache or download
    bulk_download_info_cache_path = bulk_download_path.with_suffix('.info.parquet')
    try:
        if recompute:
            raise IOError
        units_download_info_df = from_parquet(bulk_download_info_cache_path)
    except IOError:
        # Parallel download each unit metadata
        n_jobs = effective_n_jobs(n_jobs) if n_jobs is not None else 64
        units_download_info_df: pd.DataFrame = pd.concat(
            Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_download_units_info)(
                    urls=list(urls_chunk),
                    job_id=job_id,
                    add_study_metadata=True,
                    verbose=True
                )
                for job_id, urls_chunk in enumerate(distribute(n_jobs, urls))
            )
        ).reset_index(drop=True)
        # Massage the dataframe
        dtype = {
            k: t for k, t in {'Age': str,
                              'Unique sequences': pd.Int64Dtype(),
                              'Total sequences': pd.Int64Dtype()}.items()
            if k in units_download_info_df.columns
        }
        units_download_info_df = units_download_info_df.astype(dtype=dtype)
        # noinspection PyTypeChecker
        # Store as pickle (in cased debugging needs to happen)
        units_download_info_df.to_pickle(bulk_download_info_cache_path.with_suffix('.pickle'))
        # And as parquet, for consistency...
        to_parquet(units_download_info_df, bulk_download_info_cache_path)

    assert len(urls) == len(units_download_info_df)

    if not keep_missing:
        try:
            units_download_info_df = units_download_info_df[units_download_info_df['http_error'].isnull()]
        except KeyError:
            ...  # No HTTP errors

    # Make column_names play well with the likes of json
    def to_list(x):
        try:
            return list(x)
        except TypeError:
            return None
    units_download_info_df['column_names'] = units_download_info_df['column_names'].apply(to_list)

    # Remove some useless columns
    units_download_info_df = units_download_info_df.drop(columns=[
        'ETag',
        'Server',
        'Content-Type',
        'Accept-Ranges',
        'Proxy-Connection',
        'isotype'
    ])

    # Ensure unit_id is first in the frame and some other mild column reordering
    columns = ['oas_subset', 'study_id', 'unit_id']
    trailing_columns = [column for column in ('column_names', 'http_error') if column in units_download_info_df.columns]
    columns += [column for column in units_download_info_df.columns if column not in columns + trailing_columns]
    columns += trailing_columns

    return units_download_info_df[columns]


# --- Parsing ANARCI status

def _new_to_old_rule(new_rule: str) -> Tuple[str, str]:
    if new_rule == 'Missing Conserved Cysteine 23 or 104':
        return 'Missing Conserved Cysteine', ''
    if new_rule.endswith('is shorter than IMGT defined'):
        return 'Shorter than IMGT defined', new_rule.partition(' ')[0].strip()
    if new_rule.startswith('Unusual amino acid'):
        return 'Unusual residue', new_rule.split(':')[-1].strip()
    if new_rule.startswith('Insertion:'):
        return 'Insertion', new_rule.rpartition(':')[2].strip()
    raise ValueError(f'Do not know how to parse rule: "{new_rule}"')


def parse_anarci_status(status: Optional[str]) -> Dict:
    """
    Parses the ANARCI status string in OAS and returns a dictionary with all violated QA tests.

    From https://onlinelibrary.wiley.com/doi/10.1002/pro.4205
    ---
    For each sequence, the IMGT numbering scheme was added using
    antibody numbering and antigen receptor classification (ANARCI) April 23, 2020.
    Any sequence that ANARCI could not process was removed.
    This step predominantly removes sequences that contain a stop codon.
    An ANARCI status highlighting potential problems for each sequence is retained in the database.
    This status contains comments regarding unusual residues, lack of conserved cysteines,
    deletions and insertions outside of the CDRs, truncation of frameworks 1 or 4,
    and if the CDR3 is longer than 37 residues.
    Finally, sequences were grouped into units sharing the same metadata,
    the same chain (e.g., heavy, light, or paired), and isotype.
    ---

    See also SAAB:
    https://github.com/oxpig/saab_plus/blob/2cfbf90db8aba7da8ca900feae3ae3b250c6bf08/lib/python/saab_plus/aboss_utils/species_viability.py

    Parameters
    ----------
    status : string or None
      The status string as

    Examples
    --------
    >>> parse_anarci_status(None)
    {}
    >>> parse_anarci_status('|Deletions: 1, 2||Missing Conserved Cysteine: 23|')
    {'deletions': array([1, 2], dtype=uint8), 'missing_conserved_cysteine': array([23], dtype=uint8)}
    """

    if pd.isnull(status):
        return {}

    qas: Dict[str, Union[np.ndarray, set, list, bool]] = {}

    if status.startswith('['):
        # Like "['Missing Conserved Cysteine 23 or 104', 'fw1 is shorter than IMGT defined']"
        qas_iterator = (_new_to_old_rule(qa) for qa in literal_eval(status))
    else:
        # Like "|Deletions: 1, 2||Missing Conserved Cysteine: 23|"
        qas_iterator = (qa.split(':') if ':' in qa else (qa, '') for qa in status.split('|') if qa)

    for qa_type, qa_details in qas_iterator:
        qa_type = qa_type.strip()
        qa_details = qa_details.strip()
        if qa_type == 'Deletions':
            for deletion in qa_details.split(','):
                qas.setdefault('deletions', set()).add(int(deletion))
        elif qa_type == 'Insertion':
            for insertion in qa_details.split(','):
                qas.setdefault('insertions', set()).add(insertion.strip())
        elif qa_type == 'Missing Conserved Cysteine':
            qas['missing_conserved_cysteine'] = True
        elif qa_type == 'Shorter than IMGT defined':
            for region in qa_details.split(','):
                region = region.strip()
                region = {'fw1': 'fwr1', 'fw4': 'fwr4'}.get(region, region)
                qas[f'{region}_shorter_than_imgt_defined'] = True
        elif qa_type == 'Unusual residue':
            qas['unusual_residue'] = True
        elif qa_type == 'CDR3 is over 37 aa long':
            qas['cdr3_is_over_37_aa_long'] = True
        else:
            raise ValueError(f'Unknown QA type "{qa_type}" in ANARCI status "{status}"')

    if 'insertions' in qas:
        qas['insertions'] = sorted(qas['insertions'])

    if 'deletions' in qas:
        qas['deletions'] = np.array(sorted(qas['deletions']), dtype=np.uint8)

    return qas


# --- Parse OAS CSV units

def _read_unit_metadata(url_or_path, add_column_names=True, num_retries=5):
    error = None
    for retry in range(num_retries):
        try:
            with open(url_or_path) as reader:
                metadata = parse_oas_metadata_json(next(reader))
                if add_column_names:
                    metadata['column_names'] = [column.strip() for column in next(reader).split(',')]
                return metadata
        except HTTPError as ex:
            error = ex
    return {'http_error': str(error)}


def parse_oas_metadata_json(metadata_json):
    metadata_json = metadata_json.strip()
    if metadata_json.startswith('"'):
        #
        # As of 2021/09/10, unit metadata json string is overquoted
        #   Double quotes for keys and values
        #   Spurious quotes at the beginning and the end of the string
        # This fixes it, we should also report together with the other problems
        #
        metadata_json = metadata_json.strip().replace('""', '"')[1:-1]
    return json.loads(metadata_json)


ANARCI_IMGT_CDR_LENGTHS = {
    #
    # Notes:
    #
    #   - CDR1 and CDR2 lengths shall be imposed by germline and so not exceed their observed limits
    #     (e.g., IMGT bounds). But ANARCI sometimes reports longer CDRs
    #     (with the famous 62A bug, for example) and so we can flag problematic Abs (see explanations above)
    #
    #   - CDR3: IMGT really poses no constraints to the length of CDR3
    #       http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    #     But ANARCI claims it to be 65, and in turn to be able to recognize only shorter CDR3s.
    #     However, reading some code:
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/lib/python/anarci/schemes.py#L421-L436
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/build/lib/anarci/schemes.py#L485-L491
    #     Length seems limited by the insertions alphabet (52 insertion codes + space for "no insertion"):
    #       https://github.com/oxpig/ANARCI/blob/fd2f694c2c45033f356fb7077b866e3a62acfc7c/build/lib/anarci/schemes.py#L82
    #       ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
    #        "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA", "BB", "CC", "DD", "EE", "FF", "GG",
    #        "HH", "II", "JJ", "KK", "LL", "MM", "NN", "OO", "PP", "QQ", "RR", "SS", "TT", "UU",
    #        "VV", "WW", "XX", "YY", "ZZ", " "]
    #     In practice we likely cannot at this stage detect unsupported CDR3 lengths, as these shall be filtered out
    #     by ANARCI / the data processing pipeline / nature.
    #     Still let's check.

    #
    # CDR length limits, both inclusive
    # Actually let's use ANARCI output...
    #

    'cdr1': (0, 12),  # N.B. ignore IMGT lower bound of 5
    'cdr2': (0, 10),
    'cdr3': (0, 106),  # N.B. ANARCI limit 53 insertion codes x 2
}


def _preprocess_anarci_data(numbering_data_dict,
                            locus,
                            *,
                            expected_cdr3=None,
                            anarci_status=None) -> dict:
    """
    Parses the ANARCI imgt annotations in the original OAS units into a more efficient representation.
    Flags potential problems.
    """

    #
    # Explaining what follows.
    #
    # OAS was IMGT annotated with ANARCI 2020.04.23
    #   http://www.imgt.org/IMGTindex/numbering.php
    #
    # Theoretically IMGT insertions in CDR3 should only happen in residues 111 and 112
    # In 112 they follow an inverse order (i.e., B < A)
    # Like: 111, 111A, 111B, 112B, 112A, 112
    # See:
    #   https://github.com/oxpig/ANARCI/blob/master/lib/python/anarci/schemes.py
    #   http://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    #   http://www.imgt.org/IMGTScientificChart/Nomenclature/IMGTmutation.html
    #   http://www.imgt.org/FAQ/
    # In practice, we are dealing with the unexpected due to incomplete coverage
    # of existing Ab variants and bioinformatics / software problems.
    #
    # These are the known ANARCI bugs that might affect some IMGT positions:
    #
    #   - Heavy 61A before 61:
    #     https://github.com/oxpig/ANARCI/issues/14
    #     This seems actually a corner case when a CDR2 is longer than allowed by the scheme's 10 max length?
    #
    #   - Kappa 21 spurious gap (only newer ANARCIs):
    #     https://github.com/oxpig/ANARCI/issues/17
    #     We could bisect, or check one of several forks seemingly dealing with new germlines
    #
    # Plus other relevant ANARCI bugs:
    #   - Useless checking for too long CDRs
    #     https://github.com/oxpig/ANARCI/issues/8
    #     See also first bug above
    #
    # Some sanity checks stored as flags:
    #
    #   - insertion codes in positions other than 111/112 (e.g. 61A bug)
    #
    #   - too long CDRs (related to previous)
    #     Gabi's take:
    #       > Hmm... so, the V segments contain CDRs 1 and 2, so these shouldn't get longer
    #         than germline - are you looking at a different species antibody where you're seing this?
    #         If so, I suppose they would use the same .1 etc scheme that they use for CDR3, although
    #         they were probably assuming at the time that no longer loops existed...
    #         I think they included everything they knew at the time, but surely there are some odd ruminants
    #         that have freak antibodies.
    #         Are those datasets human? Could also be some experimental oddity, although I don't recall
    #         seeing changes in sequence length...
    #         If you can extract the sequences, you could see whether it's including some framework into the CDR
    #         - at least CDR1 is pretty easy to recognize
    #
    #   - not conserved cysteines
    #     Gabi's take:
    #       > first - yes, I think they mean absence of cysteins in IMGT positions 23 and/or 104.
    #         You probably can crete antibodies without these disulfides, and some people have successfully
    #         expressed antibodies in E. coli, which aren't able to form disulfide bonds, but typically
    #         those anitbodies aren't very good. In NGS data from an artificial library, I would absolutely
    #         kick sequences without either of the Cys, because it's most likely an artifact of some sort;
    #         if it's a natrual repertoire, it may be real (or a sequencing artifact, which isn't super rare in NGS),
    #         but I'd still rather not work with that antibody
    #     Note, we could extend the check to other conserved features. From the IMGT page:
    #        > The IMGT unique numbering has been defined to compare the variable domains whatever
    #          the antigen receptor, the chain type, or the species [1-3]. In the IMGT unique numbering,
    #          the conserved amino acids always have the same position, for instance cysteine 23 (1st-CYS),
    #          tryptophan 41 (CONSERVED-TRP), hydrophobic amino acid 89, cysteine 104 (2nd-CYS),
    #          phenylalanine or tryptophan 118 (J-PHE or J-TRP).
    #     But AFAICT this does not have support in the literature, as opposed to the 1st and 2nd cysteine checks
    #     See:
    #       https://doi.org/10.1101/2021.01.08.425894
    #       https://doi.org/10.4049/jimmunol.1800669
    #
    #   - wrong sequence reconstruction
    #

    #
    # Slow, but since we run only once...
    #

    regions = (
        (f'fw{locus.lower()}1', 'fwr1'),
        (f'cdr{locus.lower()}1', 'cdr1'),
        (f'fw{locus.lower()}2', 'fwr2'),
        (f'cdr{locus.lower()}2', 'cdr2'),
        (f'fw{locus.lower()}3', 'fwr3'),
        (f'cdr{locus.lower()}3', 'cdr3'),
        (f'fw{locus.lower()}4', 'fwr4'),
    )

    # We will populate all these fields for the current record
    alignment_data = {
        # alignment
        'sequence_aa': [],
        'imgt_positions': [],
        'imgt_insertions': [],
        'fwr1_start': None,
        'fwr1_length': None,
        'cdr1_start': None,
        'cdr1_length': None,
        'fwr2_start': None,
        'fwr2_length': None,
        'cdr2_start': None,
        'fwr3_start': None,
        'fwr3_length': None,
        'cdr2_length': None,
        'cdr3_start': None,
        'cdr3_length': None,
        'fwr4_start': None,
        'fwr4_length': None,
        # flags
        'has_insertions': False,
        'has_unexpected_insertions': False,
        'has_mutated_conserved_cysteines': False,
        'has_wrong_cdr3_reconstruction': None,
        'has_kappa_gap_21': False,
        'anarci_deletions': None,
        'anarci_insertions': None,
        'anarci_missing_conserved_cysteine': False,
        'anarci_unusual_residue': False,
        'anarci_fwr1_shorter_than_imgt_defined': False,
        'anarci_fwr4_shorter_than_imgt_defined': False,
        'anarci_cdr3_is_over_37_aa_long': False,
    }

    expected_keys = set(region_key for region_key, _ in regions)
    if not (expected_keys & set(numbering_data_dict)):
        raise ValueError(f'ANARCI dictionary does not contain any expected key '
                         f'({sorted(expected_keys)} vs {sorted(numbering_data_dict)}))')

    last_region_end = 0
    for region_key, region in regions:

        # What AAs do we have in the region?
        aas_in_region = list(numbering_data_dict.get(region_key, {}).items())

        # Start and length (None and 0 for not present regions)
        alignment_data[f'{region}_start'] = last_region_end if len(aas_in_region) else None
        alignment_data[f'{region}_length'] = len(aas_in_region)
        last_region_end += len(aas_in_region)

        # Sort the region AAs
        region_positions = []
        for position, aa in aas_in_region:
            position, insertion = parse_anarci_position(position)
            insertion_code = anarci_insertion_to_code(insertion)
            # Got insertions
            if insertion_code:
                alignment_data['has_insertions'] = True
                if position not in (111, 112):
                    alignment_data['has_unexpected_insertions'] = True
            if position in (23, 104) and aa != 'C':
                alignment_data['has_mutated_conserved_cysteines'] = True
            if locus == 'K' and position == 21 and aa == '-':
                alignment_data['has_kappa_gap_21'] = True
            # Mirroring of inserted residues
            insertion_code_order = insertion_code if position not in (112, 62) else -insertion_code
            region_positions.append((position, insertion_code_order, insertion, aa))
        region_positions = sorted(region_positions)

        alignment_data['sequence_aa'] += [aa for *_, aa in region_positions]
        alignment_data['imgt_positions'] += [position for position, *_ in region_positions]
        alignment_data['imgt_insertions'] += [insertion for *_, insertion, _ in region_positions]

    #
    # Make the alignment data a tad more efficient to work with. Still playing...
    #
    #   - Likely using arrays for aligned sequences is not needed.
    #     They currently precludes parquet from better representation.
    #
    #   - Probably sparse insertion codes make more sense performance-wise.
    #     They are less practical though.
    #
    alignment_data['sequence_aa'] = ''.join(alignment_data['sequence_aa'])  # use only needed bytes or S1 dtype?
    alignment_data['imgt_positions'] = np.array(alignment_data['imgt_positions'], dtype=np.dtype('u1'))
    alignment_data['imgt_insertions'] = (
        alignment_data['imgt_insertions']  # use only needed bytes / S2 dtype / int16 with mapped code?
        if alignment_data['has_insertions'] else None)

    # Check expectations
    if expected_cdr3 is not None:
        if alignment_data['cdr3_start'] is None:
            alignment_data['has_wrong_cdr3_reconstruction'] = True
        else:
            cdr3_start = alignment_data['cdr3_start']
            cdr3_end = (
                    alignment_data['cdr3_start'] + alignment_data['cdr3_length'])
            # noinspection PyUnresolvedReferences
            anarci_cdr3 = (
                alignment_data['sequence_aa'][cdr3_start:cdr3_end])
            alignment_data['has_wrong_cdr3_reconstruction'] = anarci_cdr3 != expected_cdr3

    # Add ANARCI QA flags
    if anarci_status is not None:
        anarci_status = {
            f'anarci_{qa_name}': qa_value
            for qa_name, qa_value in parse_anarci_status(anarci_status).items()
        }
        alignment_data.update(anarci_status)

    return alignment_data


def _igblast_tf_to_bool(tf):
    """Cast IGBLAST QA value to bool."""
    if pd.isnull(tf):
        return tf
    if tf == 'T':
        return True
    if tf == 'F':
        return False
    raise ValueError(f'Unknown IGBLAST value {tf}')


def _process_sequences_df(df: pd.DataFrame,
                          unit: Unit,
                          verbose=False,
                          drop_anarci_status=True) -> Tuple[pd.DataFrame, dict]:

    # --- basic sanity checks

    if unit.oas_subset == 'paired' and ('locus_heavy' not in df.columns or 'locus_light' not in df.columns):
        raise ValueError(f'Paired unit {unit.id} does not have correct locus columns')

    if unit.oas_subset == 'unpaired' and 'locus' not in df.columns:
        raise ValueError(f'Unaired unit {unit.id} does not have correct locus columns')

    if unit.oas_subset == 'unpaired':
        loci = df.locus.unique()
        if len(loci) != 1:
            raise ValueError(f'More than one locus ({loci}) in unit {unit.path}')

    # --- do munging

    # We will keep just a few of the many columns for the time being.
    # Note, nucleotides might be interested later on, together with some of the IgBlast gathered info.
    # If so, beware the size of parquet files is much larger
    # (as compression is per column and cannot realize the redundancy between columns).
    # If bringing back, the best would be to remove redundancy where possible
    # (e.g., by storing indices instead of substrings).
    # Here an example record with selected values, for documentations
    SELECTED_UNPAIRED_RECORD_EXAMPLE = {
        'locus': 'H',
        'stop_codon': 'F',
        'vj_in_frame': 'T',
        'v_frameshift': 'F',
        'productive': 'T',
        'rev_comp': 'T',
        'complete_vdj': 'F',
        'v_call': 'IGHV4-31*03',
        'd_call': 'IGHD2-8*01',
        'j_call': 'IGHJ4*02',
        'junction_aa': 'CARDTRGVGAAWSKVYW',
        f'junction_aa_length': 17.0,
        'cdr3_aa': 'WHATEVER',
        'Redundancy': 1,
        'ANARCI_numbering': "{'fwh1': {'18 ': 'T', '19 ': 'L', '20 ': ...",
        'ANARCI_status': '|Deletions: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 73, 128'
                         '|Insertion: F40A|'
                         '|Shorter than IMGT defined: fw1, fw4|'
    }

    start = time.time()

    dfs = []

    for chain_suffix in ('', '_heavy', '_light'):

        # Select chain columns
        possible_columns = [f'{column}{chain_suffix}'
                            for column in SELECTED_UNPAIRED_RECORD_EXAMPLE
                            if f'{column}{chain_suffix}' in df.columns]
        chain_df = df[possible_columns]
        if chain_df.empty:
            continue

        if verbose:
            print(f'PARSING {len(df)} RECORDS (unit={unit.id} chain_suffix={chain_suffix})')

        # Remove column chain suffix
        if chain_suffix:
            chain_df = chain_df.rename(columns=lambda column: column.rpartition('_')[0])

        # Process all the records
        processed_records = []
        for index, numbering, anarci_status, cdr3_aa, locus in zip(chain_df.index,
                                                                   chain_df[f'ANARCI_numbering'],
                                                                   chain_df[f'ANARCI_status'],
                                                                   chain_df[f'cdr3_aa'],
                                                                   chain_df[f'locus']):

            record = {'index_in_unit': index, 'chain': 'heavy' if locus == 'H' else 'light'}

            # Parse numbering
            numbering = _preprocess_anarci_data(literal_eval(numbering),
                                                locus=locus,
                                                expected_cdr3=cdr3_aa,
                                                anarci_status=anarci_status)
            record.update(numbering)

            # Add to processed records
            processed_records.append(record)

        # Merge the processed records with the selected unit metadata
        dfs.append(chain_df.merge(pd.DataFrame(processed_records),
                                  left_index=True,
                                  right_on='index_in_unit',
                                  how='left',
                                  validate='one_to_one'))

    # Concatenate all the records
    df: pd.DataFrame = pd.concat(dfs).sort_values('index_in_unit')

    # Drop some more redundant columns
    df = df.drop(columns=['cdr3_aa', 'ANARCI_numbering'])

    # Snake-casing more columns
    df = df.rename(columns={'Redundancy': 'redundancy', 'ANARCI_status': 'anarci_status'})

    # Ensure good types
    for column in ('junction_aa_length',
                   'fwr1_start',
                   'fwr1_length',
                   'cdr1_start',
                   'cdr1_length',
                   'fwr2_start',
                   'fwr2_length',
                   'cdr2_start',
                   'cdr2_length',
                   'fwr3_start',
                   'fwr3_length',
                   'cdr3_start',
                   'cdr3_length',
                   'fwr4_start',
                   'fwr4_length'):
        if column not in df.columns:
            df[column] = None
        df[column] = df[column].astype(pd.UInt16Dtype())
    if 'redundancy' not in df.columns:
        df['redundancy'] = None
    df['redundancy'] = df['redundancy'].astype(pd.UInt32Dtype())
    for column in ('index_in_unit',):
        df[column] = df[column].astype(pd.UInt64Dtype())
    for column in ('stop_codon',
                   'vj_in_frame',
                   'v_frameshift',
                   'productive',
                   'rev_comp',
                   'complete_vdj'):
        if column not in df.columns:
            df[column] = None
        else:
            df[column] = df[column].apply(_igblast_tf_to_bool)
        df[column] = df[column].astype(pd.BooleanDtype())

    #
    # TODO: keep making it a bit closer to AIRR
    # see:
    #   https://docs.airr-community.org/en/stable/
    #   https://docs.airr-community.org/en/stable/datarep/rearrangements.html
    #   https://docs.airr-community.org/en/stable/datarep/rearrangements.html#rearrangementschema
    # e.g.
    #   cdr3_start, cdr3_end -> based on 1-numbering and inclusive... (a bit inconvenient in python land)
    #   redundancy -> duplicate_count
    #

    COLUMNS = {
        # index in CSV file
        'index_in_unit': None,
        # chain + locus
        'chain': None,
        'locus': None,
        # germlines
        'v_call': None,
        'd_call': None,
        'j_call': None,
        # sequence
        'sequence_aa': None,
        # numbering
        'imgt_positions': None,   # Not sure if there is anything similar from AIRR
        'imgt_insertions': None,  # Not sure if there is anything similar from AIRR
        # is the alignment based on the opposite strand?
        'rev_comp': None,
        # junction
        'junction_aa': None,
        'junction_aa_length': None,
        # regions (at the moment, python intervals)
        'fwr1_start': None,
        'fwr1_length': None,
        'cdr1_start': None,
        'cdr1_length': None,
        'fwr2_start': None,
        'fwr2_length': None,
        'cdr2_start': None,
        'fwr3_start': None,
        'fwr3_length': None,
        'cdr2_length': None,
        'cdr3_start': None,
        'cdr3_length': None,
        'fwr4_start': None,
        'fwr4_length': None,
        # duplicate counts (N.B., likely this is "duplicate_count" in AIRR standard, but defer renaming)
        'redundancy': None,
        # igblast / airr flags
        'stop_codon': None,
        'vj_in_frame': None,
        'v_frameshift': None,
        'productive': None,
        'complete_vdj': None,
        # our flags
        'has_insertions': None,
        'has_unexpected_insertions': None,
        'has_mutated_conserved_cysteines': None,
        'has_wrong_cdr3_reconstruction': None,
        'has_kappa_gap_21': None,
        # anarci flags
        'anarci_deletions': None,
        'anarci_insertions': None,
        'anarci_missing_conserved_cysteine': None,
        'anarci_unusual_residue': None,
        'anarci_fwr1_shorter_than_imgt_defined': None,
        'anarci_fwr4_shorter_than_imgt_defined': None,
        'anarci_cdr3_is_over_37_aa_long': None,
        'anarci_status': None,
    }
    if set(COLUMNS) - set(df.columns):
        raise Exception(f'Some expected columns are missing {sorted(set(COLUMNS) - set(df.columns))} (unit={unit.id})')
    if set(df.columns) - set(COLUMNS):
        raise Exception(f'Some columns are spurious {sorted(set(df.columns) - set(COLUMNS))} (unit={unit.id})')
    renamer = {column: f'{column if new_name is None else new_name}' for column, new_name in COLUMNS.items()}
    df = df.rename(columns=renamer)[list(renamer.values())]

    # Drop anarci status?
    if drop_anarci_status:
        del df['anarci_status']

    # Done
    logs = {
        'num_records': len(df),
        'taken_s': time.time() - start
    }

    if verbose:
        print(f'DONE PARSING {len(df)} RECORDS IN {logs["taken_s"]:.2f} SECONDS')

    return df, logs


def _process_oas_csv_unit(unit: Unit,
                          parallel: Parallel = None,
                          chunk_size: int = 5_000,
                          async_io: bool = True,
                          verbose: bool = False,
                          reraise: bool = False) -> Tuple[Dict, Optional[pd.DataFrame]]:
    """
    Parses an OAS unit in the new CSV format.

    References
    ----------

    Old OPIG blogpost:
      https://www.blopig.com/blog/2020/07/observed-antibody-space-miairr/

    MiAIRR Standards:
      https://docs.airr-community.org/en/stable/miairr/introduction_miairr.html

    igblast and anarci output specs
    """

    processing_logs = {'io_waiting': 0}

    csv_chunk_reader = None
    mutex = threading.Lock()
    stop_reading_flag = []

    try:

        # N.B. do not move this to a context manager or ensure the Thread gets properlly cleaned up
        csv_chunk_reader = open(unit.original_csv_path)

        # Ignore metadata
        next(csv_chunk_reader)

        # --- Threaded manual async I/O to avoid the workers to wait for data as much as possible

        df_queue = queue.Queue(maxsize=-1)

        def queue_get():
            get_start = time.time()
            chunk = df_queue.get()
            if verbose:
                print(f'I/O QUEUE SIZE {df_queue.qsize()} UNIT={unit.original_csv_path}')
            processing_logs['io_waiting'] += time.time() - get_start
            return chunk

        def chunk_producer():
            start = time.time()
            try:
                for records in pd.read_csv(csv_chunk_reader, sep=',', low_memory=True, chunksize=chunk_size):
                    df_queue.put(records)
                    mutex.acquire()
                    if stop_reading_flag:
                        mutex.release()
                        processing_logs['io_early_stop'] = True
                        break
                    mutex.release()
            except SystemExit as ex:
                print(f'WARNING: the CSV READER HAS BEEN CLOSED ABRUPTLY')
                processing_logs['io_exception'] = str(ex)
            df_queue.put(None)
            processing_logs['taken_read_s'] = time.time() - start

        csv_reader_thread = threading.Thread(target=chunk_producer, daemon=True)
        csv_reader_thread.start()

        def chunk_consumer():
            while (batch := queue_get()) is not None:
                yield batch

        start = time.time()

        chunks = chunk_consumer()
        if not async_io:
            chunks = list(chunk_consumer())

        # --- Do the actual work

        if parallel is None:
            dfs_logs = [_process_sequences_df(df=records, unit=unit, verbose=verbose) for records in chunks]
        else:
            dfs_logs = parallel(
                delayed(_process_sequences_df)
                (df=records, unit=unit, verbose=verbose)
                for records in chunks
            )

        # Keep around also worker logs
        processing_logs['worker_logs'] = [worker_log for _, worker_log in dfs_logs]

        # In case merging fails, give a rough estimation of these...
        processing_logs['num_records'] = sum(len(df) for df, _ in dfs_logs if df is not None)
        processing_logs['taken_process_sequences_s'] = time.time() - start  # In case concat fails

        # Combine dataframes
        df = pd.concat([df for df, _ in dfs_logs]).reset_index(drop=True)
        processing_logs['num_records'] = len(df)
        processing_logs['taken_process_sequences_s'] = time.time() - start

        if verbose:
            print(f'Processed {len(df)} records in {processing_logs["taken_process_sequences_s"]:.2f} seconds '
                  f'({len(df) / processing_logs["taken_process_sequences_s"]:.2f} records/s)')

        return processing_logs, df

    except Exception as ex:
        if reraise:
            raise
        print(f'ERROR PROCESSING UNIT {unit.id}')
        traceback.print_exc()
        processing_logs['error'] = ex
        return processing_logs, None
    finally:
        # First stop the reading thread
        mutex.acquire()
        stop_reading_flag.append(True)
        mutex.release()
        # Then close the input stream
        if csv_chunk_reader is not None:
            csv_chunk_reader.close()


# --- Entry points

def cache_units_meta(recompute: bool = False, paired: bool = None, n_jobs=-1):
    """
    Caches the OAS units metadata from the web and shows some information about them.

    Parameters
    ----------
    recompute : bool, default False
      If True, recompute caches by downloading all the metadata from the online OAS units.

    paired : bool or None, default None
      If True, only report the paired subset.
      If False, only report the paired subset.
      If None, report both subsets

    n_jobs: int, default -1
      Number of jobs to use (joblib semantics)
    """

    # --- Load all units metadata and show schema

    df = oas_units_meta(recompute=recompute, paired=paired, keep_missing=True, n_jobs=n_jobs)
    df.info()

    # --- Report and get rid of downloads that are missing

    try:
        missing_downlods = df[~df.http_error.isnull()]
        df = df[df.http_error.isnull()]
        if len(missing_downlods):
            print(f'Missing downloads: {len(missing_downlods)} ({sorted(missing_downlods.url)})')
    except (KeyError, AttributeError):
        ...

    # --- Simple size statistics, compare with reported numbers in the web

    # Update these expected values with the data that appears in the webapp
    # Updated 2021/09/10
    #
    # "Your search yielded 1,535,831,757 unique sequences from 80 studies."
    #   http://opig.stats.ox.ac.uk/webapps/oas/oas
    # Of course, this is before de-duplication
    EXPECTED_UNPAIRED_NUM_STUDIES = 80
    EXPECTED_UNPAIRED_NUM_SEQUENCES = 1_535_831_757
    #
    # "Your search yielded 121,838 filtered sequences from 5 studies."
    #   http://opig.stats.ox.ac.uk/webapps/oas/oas_paired
    # Of course, this is before de-duplication
    EXPECTED_PAIRED_NUM_STUDIES = 5
    EXPECTED_PAIRED_NUM_SEQUENCES = 121_838

    num_paired_studies = df.query('oas_subset == "paired"')['study_id'].nunique()
    num_unpaired_studies = df.query('oas_subset == "unpaired"')['study_id'].nunique()
    print(f'Paired studies {num_paired_studies} (expected {EXPECTED_PAIRED_NUM_STUDIES})')
    print(f'Unpaired studies {num_unpaired_studies} (expected {EXPECTED_UNPAIRED_NUM_STUDIES})')

    print(f'Expected number of paired sequences: {EXPECTED_PAIRED_NUM_SEQUENCES}')
    print(f'Unpaired number of unpaired sequences: {EXPECTED_UNPAIRED_NUM_SEQUENCES}')
    print(df.groupby('oas_subset')[['Unique sequences', 'Total sequences']].sum())
    print('(Disagreements are likely due to URLs in the download script actually missing)')

    # --- Summarize original unit sizes

    print('File size stats (in MiB)')

    def to_mib(value):
        if not pd.isnull(value):
            return int(value) / 1024 ** 2
        return value
    df['file_size_MiB'] = df["Content-Length"].apply(to_mib)
    print(f'Total download size: {df["file_size_MiB"].sum()} MiB')
    print(df['file_size_MiB'].describe())
    with pd.option_context('display.max_rows', None):
        print(df[['url', 'file_size_MiB']].sort_values('file_size_MiB', ascending=False))

    # --- Summarize column presence

    print('Number of units each column appears in...')
    columns = []
    for _, row in df.iterrows():
        for column in row['column_names']:
            columns.append({
                'column': column,
                'subset': row['oas_subset'],
                'study_id': row['study_id'],
                'file_name': row['file_name'],
            })
    columns_df = pd.DataFrame(columns)
    columns_df.info()
    with pd.option_context('display.max_rows', None):
        print(columns_df.groupby(['subset', 'column']).size())


def download_units(oas_path=None,
                   clean_not_in_meta: bool = False,
                   ignore_with_caches: bool = False,
                   n_jobs: int = 1,
                   update: bool = False,
                   dry_run: bool = False,
                   no_drop_caches: bool = False,
                   no_resume: bool = False):
    """
    Download OAS units.

    This should be run after `cache_units_meta`.
    """

    # Once we have downloaded the metadata from the web we can use our OAS abstraction
    if oas_path is None:
        oas_path = find_oas_path()
    oas = OAS(oas_path=oas_path)

    # Remove from disk units that do not appear in metadata
    if clean_not_in_meta:
        oas.remove_units_not_in_meta(dry_run=False)

    # Collect the units to download
    units_to_download: List[Unit] = []
    for unit in oas.units_in_meta():
        if ignore_with_caches and unit.has_sequences:
            continue
        if update or unit.needs_redownload:
            units_to_download.append(unit)

    size_to_download = sum(unit.online_csv_size_bytes for unit in units_to_download)
    print(f'Downloading {len(units_to_download)} units ({size_to_download / 1024**2:.2f}MiB)')

    # Do the download
    Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(lambda unit: unit.download(force=update,
                                           dry_run=dry_run,
                                           drop_caches=not no_drop_caches,
                                           resume=not no_resume))
        (unit=unit)
        for unit in sorted(units_to_download)
    )

    # Done!
    print(f'DONE downloading {len(units_to_download)} units')


def process_units(*,
                  oas_path=None,
                  shard=0,
                  n_shards=1,
                  n_jobs=1,
                  unstable_shards=False,
                  chunk_size=5_000,
                  verbose=False,
                  recompute=False):
    """Processes CSV units to make them more digestible for analysis and learning."""

    oas = OAS(oas_path)

    # Collect and balance work to do
    units = list(oas.units_in_meta())
    if unstable_shards:
        units = [unit for unit in units if unit.should_recompute(force=recompute)]

    sizes_units = [(unit.original_csv_path.stat().st_size if unit.has_original_csv else 0, unit) for unit in units]
    sizes_units = sorted(sizes_units, reverse=True)
    total_size_mb = sum([size for size, _ in sizes_units]) / 1024**2
    sizes_units = [(size, unit) for size, unit in sizes_units[shard::n_shards]
                   if unit.should_recompute(force=recompute)]
    shard_size_mb = sum([size for size, _ in sizes_units]) / 1024**2
    print(f'Processing {shard_size_mb:.2f}MiB of {total_size_mb:.2f}MiB worth of CSVs')

    # Better actually use random order per shard, for a better time estimate
    # Even better, maybe, would be to alternate large with small units
    sizes_units = list(np.random.RandomState(seed=42).permutation(sizes_units))

    n_jobs = effective_n_jobs(n_jobs)
    processed_size = 0
    with Parallel(n_jobs=n_jobs, backend='loky', pre_dispatch=n_jobs) as parallel:
        for i, (size, unit) in enumerate(sizes_units):
            start = time.time()
            logs = {'start_time': str(datetime.datetime.now())}
            try:
                size /= 1024 ** 2
                print(f'PROCESSING unit={unit.id} size={size:.2f} MiB {i + 1} of {len(sizes_units)}')
                process_logs, df = _process_oas_csv_unit(unit=unit,
                                                         parallel=parallel,
                                                         chunk_size=chunk_size,
                                                         verbose=verbose)
                logs.update(process_logs)
                if df is not None:
                    compress_sequences_df(df=df, path=unit.sequences_path)
                else:
                    raise Exception('Workers error')
            except Exception as ex:
                logs['error'] = ex
            finally:
                logs['end_time'] = str(datetime.datetime.now())
                logs['taken_s'] = time.time() - start
                pd.to_pickle(logs, unit.sequences_path.with_suffix('.processing-logs.pickle'))
                if logs.get('error') is not None:
                    pd.to_pickle(logs['error'], unit.sequences_path.with_suffix('.processing-error.pickle'))
                processed_size += size
                print(f'PROCESSED unit={unit.id} size={size:.2f} MiB {i + 1} of {len(sizes_units)} '
                      f'({processed_size:.2f} of {shard_size_mb:.2f} MiB) '
                      f'in {logs["taken_s"]:.2f}s')


def _processing_clis():
    num_jobs = 8
    chunk_size = 8000
    machine_shards = {
        # 'dgx1': 8,
        'dgx2': 40,
        # 'dgx3': 10,
        # 'dgx4': 28,
    }
    total_shards = sum(machine_shards.values())
    machines = list(chain(*[[machine] * num_shards for machine, num_shards in machine_shards.items()]))
    assert len(machines) == total_shards
    machines = np.random.RandomState(seed=37).permutation(machines)

    commands = {}
    for shard_num, machine in enumerate(machines):
        command = (f'oas process-units '
                   f'--shard {shard_num} '
                   f'--n-shards {total_shards} '
                   # f'--unstable-shards '
                   f'--n-jobs {num_jobs} '
                   f'--chunk-size {chunk_size} '
                   f'&>shard-{shard_num}-{total_shards}-{machine}.log &')
        commands.setdefault(machine, []).append(command)
        commands[machine].append('disown')

    chores_path = Path(__file__).parent.parent.parent / 'chores'
    chores_path.mkdir(exist_ok=True, parents=True)
    for machine, commands in commands.items():
        with open(chores_path / f'{machine}-process-units.sh', 'wt') as writer:
            writer.write('\n'.join(commands))


def parse_all_anarci_status():
    """
    Iterate over all anarci status and parse them.

    This is to find unsupported rules, as there are no specs for these strings.
    """
    oas = OAS()
    for unit in sorted(oas.units_in_meta()):
        print(unit.id)
        if unit.has_sequences:
            df = unit.sequences_df()
            try:
                for status in df.anarci_status:
                    try:
                        parse_anarci_status(status)
                    except ValueError as ex:
                        if 'QA' in str(ex):
                            print(str(ex))
                        else:
                            print(f'Exception for {status}: {str(ex)}')
            except AttributeError:
                ...


# --- Some things that should become proper tests

def check_csv_parsing_corner_cases():
    # Very long CDR3s with insertions, codes at and beyond "AA"
    oas = OAS()
    unit = oas.unit(oas_subset='unpaired', study_id='Kim_2020', unit_id='SRR12326757_Heavy_IGHA')
    _process_oas_csv_unit(unit, async_io=False, verbose=True, reraise=True)


def check_csv_parsing():
    oas = OAS()
    for unit in oas.units_in_disk(oas_subset='unpaired'):
        _process_oas_csv_unit(unit=unit, async_io=False, verbose=True, reraise=True)
    for unit in oas.units_in_disk(oas_subset='paired'):
        _process_oas_csv_unit(unit=unit, async_io=True, chunk_size=1, verbose=True, reraise=True)


# --- Where there is smoke...

if __name__ == '__main__':
    # Generate CLIs
    _processing_clis()

    # Smoke test local
    process_units()
    for unit in OAS().units_in_disk():
        unit.sequences_df().info()

# --- Brain dumps

#
# The world of zlib alternatives is quite interesting:
#   - https://lemire.me/blog/2021/06/30/compressing-json-gzip-vs-zstd/
#   - http://www.htslib.org/benchmarks/zlib.html
#   - https://bugs.python.org/issue41566
#
