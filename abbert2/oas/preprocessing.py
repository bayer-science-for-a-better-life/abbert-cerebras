"""Processing the original OAS data files."""
import datetime
import json
import queue
import threading
import time
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

from abbert2.oas.common import find_oas_path
from abbert2.common import to_parquet, from_parquet
from abbert2.oas.oas import OAS, Unit


def _read_unit_metadata(url_or_path, add_column_names=True):
    try:
        with open(url_or_path) as reader:
            metadata = parse_oas_metadata_json(next(reader))
            if add_column_names:
                metadata['column_names'] = [column.strip() for column in next(reader).split(',')]
            return metadata
    except HTTPError as ex:
        return {'http_error': str(ex)}


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

#
# --- Manipulation of OAS "bulk_download.sh" and unit metadata
#
# To get this file for *unpaired* data:
#   - Go to http://opig.stats.ox.ac.uk/webapps/oas/oas
#   - Click search without adding any filter
#   - Download the shell script and put it under "<oas_root>/unpaired"
#
# To get this file for *paired* data:
#   - Go to http://opig.stats.ox.ac.uk/webapps/oas/oas_paired
#   - Click search without adding any filter
#   - Download the shell script and put it under "<oas_root>/paired"
#


def _preprocess_anarci_data(numbering_data_dict, locus, *, expected_sequence=None, expected_cdr3=None) -> dict:
    """
    Parses the ANARCI imgt annotations in the original OAS units into a more efficient representation.
    Flags potential problems.
    """

    #
    # Explaining what follows.
    #
    # OAS was IMGT annotated with some (old?) version of ANARCI
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
    #   - unfit (having any of the previous)
    #

    #
    # Slow, but since we run only once...
    #

    regions = (
        (f'fw{locus.lower()}1', f'fw1'),
        (f'cdr{locus.lower()}1', f'cdr1'),
        (f'fw{locus.lower()}2', f'fw2'),
        (f'cdr{locus.lower()}2', f'cdr2'),
        (f'fw{locus.lower()}3', f'fw3'),
        (f'cdr{locus.lower()}3', f'cdr3'),
        (f'fw{locus.lower()}4', f'fw4'),
    )

    heavy_or_light = 'heavy' if locus == 'H' else 'light'

    # We will populate all these fields for the current record
    alignment_data = {
        # flags
        f'unfit_{heavy_or_light}': False,
        f'has_unexpected_insertions_{heavy_or_light}': False,
        f'has_mutated_conserved_cysteines_{heavy_or_light}': False,
        f'has_wrong_sequence_reconstruction_{heavy_or_light}': None,
        f'has_wrong_cdr3_reconstruction_{heavy_or_light}': None,
        f'has_kappa_gap_21_{heavy_or_light}': False,
        f'has_long_cdr1_{heavy_or_light}': None,
        f'has_long_cdr2_{heavy_or_light}': None,
        f'has_long_cdr3_{heavy_or_light}': None,
        # alignment
        f'has_insertions_{heavy_or_light}': False,
        f'fw1_start_{heavy_or_light}': None,
        f'fw1_length_{heavy_or_light}': None,
        f'cdr1_start_{heavy_or_light}': None,
        f'cdr1_length_{heavy_or_light}': None,
        f'fw2_start_{heavy_or_light}': None,
        f'fw2_length_{heavy_or_light}': None,
        f'cdr2_start_{heavy_or_light}': None,
        f'fw3_start_{heavy_or_light}': None,
        f'fw3_length_{heavy_or_light}': None,
        f'cdr2_length_{heavy_or_light}': None,
        f'cdr3_start_{heavy_or_light}': None,
        f'cdr3_length_{heavy_or_light}': None,
        f'fw4_start_{heavy_or_light}': None,
        f'fw4_length_{heavy_or_light}': None,
        f'aligned_sequence_{heavy_or_light}': [],
        f'positions_{heavy_or_light}': [],
        f'insertions_{heavy_or_light}': [],
    }

    last_region_end = 0
    for region_key, region in regions:

        # What AAs do we have in the region?
        aas_in_region = list(numbering_data_dict.get(region_key, {}).items())

        # Start and length (None and 0 for not present regions)
        alignment_data[f'{region}_start_{heavy_or_light}'] = last_region_end if len(aas_in_region) else None
        alignment_data[f'{region}_length_{heavy_or_light}'] = len(aas_in_region)
        last_region_end += len(aas_in_region)

        # Detect unsupported CDR lengths (we could do the same with framework regions)
        if region.startswith('cdr') and 0 < len(aas_in_region):
            alignment_data[f'has_long_{region}_{heavy_or_light}'] = (
                    len(aas_in_region) > ANARCI_IMGT_CDR_LENGTHS[region][1]
            )

        # Sort the region AAs
        region_positions = []
        for position, aa in aas_in_region:
            try:
                position, insertion_code = int(position), ' '
            except ValueError:
                position, insertion_code = int(position[:-1]), position[-1]
                # Got insertions
                alignment_data[f'has_insertions_{heavy_or_light}'] = True
                if position not in (111, 112):
                    alignment_data[f'has_unexpected_insertions_{heavy_or_light}'] = True
            if position in (23, 104) and aa != 'C':
                alignment_data[f'has_mutated_conserved_cysteines_{heavy_or_light}'] = True
            if locus == 'K' and position == 21 and aa == '-':
                alignment_data[f'has_kappa_gap_21_{heavy_or_light}'] = True
            # Mirroring of inserted residues
            insertion_code_order = ord(insertion_code) if position not in (112, 62) else -ord(insertion_code)
            region_positions.append((position, insertion_code_order, insertion_code, aa))
        region_positions = sorted(region_positions)

        alignment_data[f'aligned_sequence_{heavy_or_light}'] += [aa for *_, aa in region_positions]
        alignment_data[f'positions_{heavy_or_light}'] += [position for position, *_ in region_positions]
        alignment_data[f'insertions_{heavy_or_light}'] += [insertion for *_, insertion, _ in region_positions]

    # Make the alignment data a tad more efficient to work with. Still playing...
    #   - Likely using arrays for aligned sequences is not needed
    #     (and precludes parquet from better representation?)
    #     If so, just ''.join() both the sequence and the insertion codes.
    #   - Probably sparse insertion codes make more sense performancewise,
    #     they are less practical though.
    #   - Likely u1 dtype can work for positions (evaluate after collecting stats).
    alignment_data[f'aligned_sequence_{heavy_or_light}'] = (
        np.array(alignment_data[f'aligned_sequence_{heavy_or_light}'], dtype='S1'))
    alignment_data[f'positions_{heavy_or_light}'] = (
        np.array(alignment_data[f'positions_{heavy_or_light}'], dtype=np.dtype('u2')))
    alignment_data[f'insertions_{heavy_or_light}'] = (
        np.array(alignment_data[f'insertions_{heavy_or_light}'], dtype='S2')
        if alignment_data[f'has_insertions_{heavy_or_light}'] else None)
    if expected_sequence is not None:
        # noinspection PyUnresolvedReferences
        alignment_data[f'has_wrong_sequence_reconstruction_{heavy_or_light}'] = (
                alignment_data[f'aligned_sequence_{heavy_or_light}'].tobytes().decode('utf-8') != expected_sequence
        )
    if expected_cdr3 is not None:
        if alignment_data[f'cdr3_start_{heavy_or_light}'] is None:
            alignment_data[f'has_wrong_cdr3_reconstruction_{heavy_or_light}'] = True
        else:
            cdr3_start = alignment_data[f'cdr3_start_{heavy_or_light}']
            cdr3_end = (
                    alignment_data[f'cdr3_start_{heavy_or_light}'] + alignment_data[f'cdr3_length_{heavy_or_light}'])
            # noinspection PyUnresolvedReferences
            aligned_cdr3 = (
                alignment_data[f'aligned_sequence_{heavy_or_light}'][cdr3_start:cdr3_end].tobytes().decode('utf-8'))
            alignment_data[f'has_wrong_cdr3_reconstruction_{heavy_or_light}'] = aligned_cdr3 != expected_cdr3

    # Final veredict about the fitness of the chain
    QC_FLAGS = (
        # f'has_unexpected_insertions_{heavy_or_light}',
        f'has_mutated_conserved_cysteines_{heavy_or_light}',
        # f'has_kappa_gap_21_{heavy_or_light}',
        # f'has_wrong_sequence_reconstruction_{heavy_or_light}',
        # f'has_wrong_cdr3_reconstruction_{heavy_or_light}',
        # f'has_long_cdr1_{heavy_or_light}',
        # f'has_long_cdr2_{heavy_or_light}',
        # f'has_long_cdr3_{heavy_or_light}',
    )
    alignment_data[f'unfit_{heavy_or_light}'] = any(alignment_data[flag] for flag in QC_FLAGS)

    return alignment_data
    # FIXME: all this string manipulation is going to pass a toll


def _per_chain_columns(columns, df=None):
    if isinstance(columns, str):
        columns = [columns]
    possible_columns = [f'{column}_{chain}' for column in columns for chain in ('heavy', 'light')]
    if df is not None:
        possible_columns = [column for column in possible_columns if column in df.columns]
    return possible_columns


def _process_sequences_df(df, unit: Unit, verbose=False):

    logs = {'num_records': len(df)}

    # Make sure unpaired units get the same suffix as paired ones
    start = time.time()
    df = _rename_by_chain_type(df, unit=unit)
    logs['taken_rename_chain_type'] = time.time() - start

    # We will keep just a few of the many columns for the time being.
    # Note, nucleotides might be interested later on, together with some of the IgBlast gathered info.
    # If so, beware the size of parquet files is much larger
    # (as compression is per column and cannot realize the redundancy between columns).
    # If bringing back, the best would be to remove redundancy where possible
    # (e.g., by storing indices instead of substrings).
    # Here an example record with values, for documentations
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
    df = df[_per_chain_columns(list(SELECTED_UNPAIRED_RECORD_EXAMPLE), df=df)]
    logs['taken_select_columns'] = time.time() - start

    # Process the numbering
    # Here we also check things like if a buggy ANARCI version was used

    if verbose:
        print(f'PARSING {len(df)} RECORDS')

    for chain in ('heavy', 'light'):

        if f'ANARCI_numbering_{chain}' not in df.columns:
            continue

        records = zip(df.index,
                      df[f'ANARCI_numbering_{chain}'],
                      df[f'ANARCI_status_{chain}'],
                      df[f'cdr3_aa_{chain}'],
                      df[f'locus_{chain}'])

        processed_records = []

        start = time.time()

        for index, numbering, anarci_status, cdr3_aa, locus in records:

            record = {'index': index}

            #
            # TODO: Parse ANARCI status
            # Maybe a link from here?
            #   https://github.com/oxpig/saab_plus/blob/2cfbf90db8aba7da8ca900feae3ae3b250c6bf08/lib/python/saab_plus/aboss_utils/species_viability.py
            # Hoping to come up with some documentation
            #
            # for message in anarci_status.split('|'):
            #     if 1 < len(message):
            #         name, value = message.split(':')
            #         name = _to_snake_case(name)
            #         if name == 'deletions':
            #             ...
            #         elif name == 'insertion':
            #             ...
            #         elif name == 'shorter_than_imgt_defined':
            #             ...
            #         else:
            #             print(f'ANARCI_MESSAGE {message}')
            #

            # Parse numbering
            numbering_start = time.time()
            numbering = _preprocess_anarci_data(literal_eval(numbering),
                                                locus=locus,
                                                expected_sequence=None,  # bring back if needed
                                                expected_cdr3=cdr3_aa)

            record.update(numbering)
            processed_records.append(record)
            logs[f'numbering_{chain}_taken_s'] = time.time() - numbering_start

        merging_start = time.time()
        df: pd.DataFrame = df.merge(pd.DataFrame(processed_records).set_index('index', drop=True),
                                    left_index=True,
                                    right_index=True,
                                    how='left',
                                    validate='one_to_one')
        logs[f'merging_{chain}_taken_s'] = time.time() - merging_start

    # Drop some more redundant columns
    df = df.drop(columns=_per_chain_columns(('cdr3_aa', 'ANARCI_numbering'), df=df))

    # Type and rename a couple more columns
    for column in _per_chain_columns('junction_aa_length', df=df):
        df[column] = df[column].astype(pd.UInt16Dtype())
    df = df.rename(columns={
        'Redundancy_heavy': 'redundancy_heavy',
        'ANARCI_status_heavy': 'anarci_status_heavy',
        'Redundancy_light': 'redundancy_light',
        'ANARCI_status_light': 'anarci_status_light',
    })

    logs['taken_s'] = time.time() - start

    if verbose:
        print(f'DONE PARSING {len(df)} RECORDS IN {logs["taken_s"]:.2f} SECONDS')

    return df, logs


def _rename_by_chain_type(df, unit: Unit):
    # Is this an unpaired unit? => Name as paired (clunky but works)
    if 'sequence' in df.columns:
        loci = df.locus.unique()
        if len(loci) != 1:
            raise Exception(f'More than one locus ({loci}) in unit {unit.path}')
        suffix = '_heavy' if loci[0] == 'H' else '_light'
        df = df.rename(columns=lambda colname: colname + suffix)
    return df


def _process_oas_csv_unit(unit: Unit,
                          parallel: Parallel = None,
                          chunk_size: int = 5_000,
                          verbose: bool = False) -> Tuple[Dict, Optional[pd.DataFrame]]:
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

        # --- Do the actual work

        dfs_logs = parallel(
            delayed(_process_sequences_df)
            (df=records, unit=unit, verbose=verbose)
            for records in chunk_consumer()
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
    # These seem to correstpont to the same SRA deposition used in different studies.
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
                # TODO: can we reuse sessions / do one connection?
                # TODO: get from local cache if possible
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
    """

    if paired is None:
        # merge both paired and unpaired subsets
        subset_dfs = []
        for paired in (True, False):
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

    # Account for legacy dumps
    # FIXME: remove when dumps are updated
    if 'unit_id' not in units_download_info_df.columns:
        units_download_info_df['unit_id'] = units_download_info_df['file_name'].str.replace('.csv.gz', '', regex=False)

    # Ensure unit_id is first in the frame
    columns = ['oas_subset', 'study_id', 'unit_id']
    columns += [column for column in units_download_info_df.columns if column not in columns]

    # Make column_names play well with the likes of json
    def to_list(x):
        try:
            return list(x)
        except TypeError:
            return None
    units_download_info_df['column_names'] = units_download_info_df['column_names'].apply(to_list)

    return units_download_info_df[columns]


# --- Entry points

def cache_units_meta(recompute: bool = False, paired: bool = None):
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
    """

    # --- Load all units metadata and show schema

    df = oas_units_meta(recompute=recompute, paired=paired, keep_missing=True)
    df.info()

    # --- Report and get rid of downloads that are missing

    try:
        missing_downlods = df[~df.http_error.isnull()]
        df = df[df.http_error.isnull()]
        if len(missing_downlods):
            print(f'Missing downloads: {len(missing_downlods)} ({sorted(missing_downlods.url)})')
    except KeyError:
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
    df['file_size_MiB'] = df["Content-Length"].astype(int) / 1024 ** 2
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
                print(f'PROCESSING unit {i+1} of {len(sizes_units)} ({size:.2f} MiB)')
                process_logs, df = _process_oas_csv_unit(unit=unit,
                                                         parallel=parallel,
                                                         chunk_size=chunk_size,
                                                         verbose=verbose)
                logs.update(process_logs)
                if df is not None:
                    to_parquet(df, unit.sequences_path)
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
                print(f'PROCESSED unit {i + 1} ({unit.id}) of {len(sizes_units)} '
                      f'(total {processed_size:.2f} of {shard_size_mb:.2f} MiB) '
                      f'in {logs["taken_s"]:.2f}s')


def _processing_clis():
    num_jobs = 8
    chunk_size = 8000
    machine_shards = {
        # 'dgx1': 8,
        'dgx2': 10,
        'dgx3': 10,
        'dgx4': 28,
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


# --- Where there is smoke

if __name__ == '__main__':
    # Generate CLIs
    _processing_clis()

    # Smoke test local
    process_units()
    for unit in OAS().units_in_disk():
        unit.sequences_df().info()

# --- Brain dumps

# TODO: Manage incremental updates to the dataset, maybe also taking care of the downloads ourselves here
# TODO: even with the heavy and light complecity, still we are capturing only symmetryc dimers
# TODO: play with arrow / parquet nested structures for vh/vl (although library conveniences are best here)
# TODO: remove most HTTP headers from metadata (you lazy)

#
# The world of zlib alternatives is quite interesting:
#   - https://lemire.me/blog/2021/06/30/compressing-json-gzip-vs-zstd/
#   - http://www.htslib.org/benchmarks/zlib.html
#   - https://bugs.python.org/issue41566
#
