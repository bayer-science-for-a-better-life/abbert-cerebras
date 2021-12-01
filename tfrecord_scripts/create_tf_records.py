import os

from abbert2.oas.oas import OAS, Unit
from tfrecord_scripts.utils import (
    create_bytes_feature,
    create_int_feature,
)

from pathlib import Path
import tensorflow as tf

import json
import argparse
import sys
import numpy as np
import pandas as pd
from itertools import combinations

def create_arg_parser():
    """
    Create parser for command line args.
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_input_folder",
        required=True,
        help="src_input_folder",
    )
    parser.add_argument(
        "--out_tf_records_fldr",
        required=True,
        help="out_tf_records_fldr",
    )

    return parser


def _preprocess_df(df):

    # Drop rows with NaNs
    df = df.dropna()

    # Check for negative values ?

    # shuffle dataframe rows
    df = df.sample(frac=1)

    return df


def get_sequence(oas_path):
    unit = OAS(oas_path).units_in_path()
    return unit

def create_examples(df, metadata, *unit_id):

    feature = {}
    study_year = metadata["study_year"]
    normalized_species = metadata["normalized_species"]
    oas_subset, study_id, unit_id = unit_id

    for _, row in df.iterrows():
        try:
            feature["tokens"] = row["sequence_aa"]
            feature["cdr_start"] = [int(row[f"cdr1_start"]), int(row[f"cdr2_start"]), int(row[f"cdr3_start"])]
            feature["cdr_length"] = [int(row[f"cdr1_length"]), int(row[f"cdr2_length"]), int(row[f"cdr3_length"])]
            feature["fw_start"] = [int(row[f"fw1_start"]), int(row[f"fw2_start"]), int(row[f"fw3_start"]), int(row[f"fw4_start"])]
            feature["fw_length"] = [int(row[f"fw1_length"]), int(row[f"fw2_length"]), int(row[f"fw3_length"]), int(row[f"fw4_length"])]
            feature["chain"] = row["chain"]
            feature["index_in_unit"] = row["index_in_unit"]

            feature["study_year"] = study_year
            feature["oas_subset"] = oas_subset
            feature["normalized_species"] = normalized_species
            feature["unit_id"] = unit_id
            feature["study_id"] = study_id
            

            tf_example = create_unmasked_tokens_example(feature)

            yield (tf_example, len(feature["tokens"]))

        except Exception as e:
            print(e)
            yield None, None


def create_unmasked_tokens_example(feature):
    """
    Create tf.train.Example containing variable length sequence of tokens.
    """
    array = [create_bytes_feature(token) for token in feature["tokens"]]
    feature_lists_dict = {"tokens": tf.train.FeatureList(feature=array)}
    feature_lists = tf.train.FeatureLists(feature_list=feature_lists_dict)

    context_features_dict = {}
    context_features_dict["study_year"] = create_int_feature([feature["study_year"]])
    context_features_dict["oas_subset"] = create_bytes_feature(feature["oas_subset"].encode())
    context_features_dict["normalized_species"] = create_bytes_feature(feature["normalized_species"].encode())
    context_features_dict["unit_id"] = create_bytes_feature(feature["unit_id"].encode())
    context_features_dict["study_id"] = create_bytes_feature(feature["study_id"].encode())
    context_features_dict["index_in_unit"] = create_int_feature(feature["index_in_unit"].encode())
    context_features_dict["chain"] = create_bytes_feature(feature["chain"].encode())
    
    context_features_dict["cdr_start"] = create_int_feature(feature["cdr_start"])
    context_features_dict["cdr_length"] = create_int_feature(feature["cdr_length"])
    context_features_dict["fw_start"] = create_int_feature(feature["fw_start"])
    context_features_dict["fw_length"] = create_int_feature(feature["fw_length"])

    context = tf.train.Features(feature=context_features_dict)
    
    tf_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return tf_example


def get_output_file_names(dest_tfrecs_fldr, unit_id, prefix):

    out_tfrecord_name = f"{prefix}_{unit_id}.tfrecord"
    out_tfrecord_name = os.path.join(dest_tfrecs_fldr, out_tfrecord_name)

    out_stats_file_name = f"{prefix}_{unit_id}_stats.json"
    out_stats_file_name = os.path.join(dest_tfrecs_fldr, out_stats_file_name)

    return out_tfrecord_name, out_stats_file_name


def partition_df(df, split, seed=1204):
    """
    split: dict with keys: train, val, test
            values  [0, 1)
    """

    heavy_df = df[df["chain"]=="heavy"]
    light_df = df[df["chain"]=="light"]

    assert len(heavy_df)+ len(light_df) == len(df)

    assert sum(list(split.values())) == 1

    print(f"len heavy: {len(heavy_df)}")
    print(f"len light: {len(light_df)}")

    heavy_mask = np.random.rand(len(heavy_df))
    light_mask = np.random.rand(len(light_df))

    prev_val = 0.0

    out = {}

    for split_type, val in split.items():
        split_heavy_mask = (heavy_mask >= prev_val) & (heavy_mask < prev_val + val) 
        split_heavy_df = heavy_df[split_heavy_mask]

        split_light_mask = (light_mask >= prev_val) & (light_mask < prev_val + val)
        split_light_df = light_df[split_light_mask]

        out[f"{split_type}_{val}_df"] = pd.concat([split_heavy_df, split_light_df])

        prev_val = val

    # Check if no overlap in train test and val data
    for combo in combinations(list(out.keys()), 2):
        key1, key2 = combo
        df_1 = out[key1]
        df_2 = out[key2]

        intersect_df = pd.merge(df_1, df_2, 'inner')
        assert intersect_df.empty

    return out

def create_tfrecords(src_input_folder, out_tf_records_fldr):
    """
    src_input_folder: ex: /cb/ml/aarti/bayer_sample/paired/Eccles_2020/SRR10358525_paired
    """
    print(f"-- src_input_folder: {src_input_folder}")
    print(f"--- out_tf_records_fldr : {out_tf_records_fldr}")
    unit = get_sequence(oas_path=src_input_folder)

    src_input_folder = Path(src_input_folder)
    *_, oas_subset, study_id, unit_id = src_input_folder.parts

    dest_tfrecs_fldr = os.path.join(out_tf_records_fldr, oas_subset, study_id, unit_id)

    if not os.path.exists(dest_tfrecs_fldr):
        os.makedirs(dest_tfrecs_fldr)
    
    if not unit.has_sequences:
        sys.exit("No dataframe sequences")

    df = unit.sequences_df()
    df = _preprocess_df(df)
    metadata = unit.nice_metadata

    split = {"train": 0.8, "val": 0.1, "test": 0.1}

    partitions = partition_df(df, split)


    for key, subset_df in partitions.items():

        # Shuffle again
        subset_df = subset_df.sample(frac=1)

        out_tfrecord_name, out_stats_file_name = get_output_file_names(dest_tfrecs_fldr, unit_id, prefix=key)

        len_df = len(subset_df)
        num_tfrecs = int(len_df // 100000) + 1
            
        print(f"---dataframe rows : {len(subset_df)}")

        dir_name, tf_fname = os.path.split(out_tfrecord_name)
        prefix = tf_fname.split(".tfrecord")[0]

        writers = []

        for i in range(num_tfrecs):
            output_file_name = prefix + f"_{i}.tfrecord"
            output_file_name = os.path.join(dir_name, output_file_name)
            writers.append(tf.io.TFRecordWriter(output_file_name))

        writer_index = 0
        num_examples = 0
        max_length = float("-inf")

        for tf_example, len_tokens in create_examples(subset_df, metadata, unit.id):

            if tf_example is not None and len_tokens is not None:
                    writers[writer_index].write(tf_example.SerializeToString())
                    writer_index = (writer_index + 1) % len(writers)
                    num_examples += 1
                    max_length = max(max_length, len_tokens)

                if num_examples % 10000 == 0:
                    print(f"--- Wrote {num_examples} examples so far ...")
            
            print(f"----DONE: {out_tfrecord_name} -  Wrote {num_examples} examples")
            for writer in writers:
                writer.close()

            with open(out_stats_file_name, "w") as stats_fh:
                json_dict = {
                    "tfrec_filename": out_tfrecord_name, 
                    "num_examples": num_examples, 
                    "max_aligned_sequence_length": max_length
                    }
                json.dump(json_dict, stats_fh)

        else:
            # File corrupted
            with open(out_stats_file_name, "w") as stats_fh:
                json_dict = {
                    "tfrec_filename": out_tfrecord_name, 
                    "num_examples": "file_corrupted or empty dataframe after filtering", 
                    "max_aligned_sequence_length": "file_corrupted or empty dataframe after filtering"
                    }
                json.dump(json_dict, stats_fh)



def main():
    """
    Main function
    """
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    create_tfrecords(args.src_input_folder, args.out_tf_records_fldr)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
    # create_tfrecords("/cb/ml/aarti/bayer_sample/paired/Eccles_2020/SRR10358525_paired", out_tf_records_fldr="/cb/ml/aarti/bayer_sample_tfrecs")
