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
import xxhash

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
    parser.add_argument(
        "--hash_partition",
        required=True,
        help="if true, use hashing to divide to subsets",
    )

    return parser


def _preprocess_df(df, seed=1204):

    # # Drop rows with NaNs
    df = df.dropna(subset=[
        "sequence_aa", 
        "cdr1_start", 
        "cdr2_start", 
        "cdr3_start",
        "cdr1_length", 
        "cdr2_length", 
        "cdr3_length",
        "fwr1_start", 
        "fwr2_start", 
        "fwr3_start", 
        "fwr4_start", 
        "fwr1_length", 
        "fwr2_length", 
        "fwr3_length", 
        "fwr4_length"])

    # Check for negative values 
    df = df[(df["cdr1_start"] >= 0) & (df["cdr2_start"] >= 0) & (df["cdr3_start"] >= 0)]
    df = df[(df["cdr1_length"] >= 0) & (df["cdr2_length"] >= 0) & (df["cdr3_length"] >= 0)]
    df = df[(df["fwr1_start"] >= 0) & (df["fwr2_start"] >= 0) & (df["fwr3_start"] >= 0) & (df["fwr4_start"] >= 0)]
    df = df[(df["fwr1_length"] >= 0) & (df["fwr2_length"] >= 0) & (df["fwr3_length"] >= 0) & (df["fwr4_length"] >= 0)]

     

    # shuffle dataframe rows
    df = df.sample(frac=1, random_state=seed)

    return df


def get_sequence(oas_path):
    unit = list(OAS(oas_path).units_in_path())
    print(f"len unit oas : {len(unit)}")
    assert len(unit) == 1
    
    return unit[0]

def create_examples(df, study_year, normalized_species, oas_subset, study_id, unit_id):

    feature = {}
    for _, row in df.iterrows():
        try:
            feature["tokens"] = row["sequence_aa"]
            feature["cdr_start"] = [int(row[f"cdr1_start"]), int(row[f"cdr2_start"]), int(row[f"cdr3_start"])]
            feature["cdr_length"] = [int(row[f"cdr1_length"]), int(row[f"cdr2_length"]), int(row[f"cdr3_length"])]
            feature["fw_start"] = [int(row[f"fwr1_start"]), int(row[f"fwr2_start"]), int(row[f"fwr3_start"]), int(row[f"fwr4_start"])]
            feature["fw_length"] = [int(row[f"fwr1_length"]), int(row[f"fwr2_length"]), int(row[f"fwr3_length"]), int(row[f"fwr4_length"])]
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
    array = [create_bytes_feature(token.encode()) for token in feature["tokens"]]
    feature_lists_dict = {"tokens": tf.train.FeatureList(feature=array)}
    feature_lists = tf.train.FeatureLists(feature_list=feature_lists_dict)

    context_features_dict = {}
    context_features_dict["study_year"] = create_int_feature([feature["study_year"]])
    context_features_dict["oas_subset"] = create_bytes_feature(feature["oas_subset"].encode())
    context_features_dict["normalized_species"] = create_bytes_feature(feature["normalized_species"].encode())
    context_features_dict["unit_id"] = create_bytes_feature(feature["unit_id"].encode())
    context_features_dict["study_id"] = create_bytes_feature(feature["study_id"].encode())
    context_features_dict["index_in_unit"] = create_int_feature([feature["index_in_unit"]])
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

    out_stats_file_name = f"{unit_id}_stats.json"
    out_stats_file_name = os.path.join(dest_tfrecs_fldr, out_stats_file_name)

    return out_tfrecord_name, out_stats_file_name


def assign_mlsubset_by_sequence_hashing(df, seed=1204, train_pct=80, validation_pct=10, test_pct=10):

    if (train_pct + validation_pct + test_pct) != 100:
        raise ValueError(f'train_pct + val_pct + test_pct must equal 100, but it is {(train_pct + validation_pct + test_pct)}')

    df["hash"] = df.apply(lambda row: xxhash.xxh32_intdigest(row.sequence_aa, seed=seed) % 100, axis=1)

    out = {}
    stats = {}

    train_df = df.query(f"hash <= {train_pct}")
    len_heavy_df = len(train_df[train_df["chain"]=="heavy"])
    len_light_df =  len(train_df[train_df["chain"]=="light"])
    out[f"train_{train_pct}"] = train_df
    stats[f"train_{train_pct}/num_heavy_sequences"] = len_heavy_df
    stats[f"train_{train_pct}/num_light_sequences"] = len_light_df
    stats[f"train_{train_pct}/num_sequences"] = len_heavy_df + len_light_df


    val_df = df.query(f"{train_pct} < hash <= {validation_pct + train_pct}")
    len_heavy_df = len(val_df[val_df["chain"]=="heavy"])
    len_light_df =  len(val_df[val_df["chain"]=="light"])
    out[f"val_{validation_pct}"] = val_df
    stats[f"val_{validation_pct}/num_heavy_sequences"] = len_heavy_df
    stats[f"val_{validation_pct}/num_light_sequences"] = len_light_df
    stats[f"val_{validation_pct}/num_sequences"] = len_heavy_df + len_light_df


    test_df = df.query(f"hash > {validation_pct + train_pct}")
    len_heavy_df = len(test_df[test_df["chain"]=="heavy"])
    len_light_df =  len(test_df[test_df["chain"]=="light"])
    out[f"test_{test_pct}"] = test_df
    stats[f"test_{test_pct}/num_heavy_sequences"] = len_heavy_df
    stats[f"test_{test_pct}/num_light_sequences"] = len_light_df
    stats[f"test_{test_pct}/num_sequences"] = len_heavy_df + len_light_df

    return out, stats


def partition_df(df, split, seed=1204):
    """
    split: dict with keys: train, val, test
            values  [0, 1)
    """

    heavy_df = df[df["chain"]=="heavy"]
    light_df = df[df["chain"]=="light"]
    heavy_df = heavy_df.sample(frac=1, random_state=seed)
    light_df = light_df.sample(frac=1, random_state=seed)

    assert len(heavy_df)+ len(light_df) == len(df)

    assert sum(list(split.values())) == 1

    print(f"-- len heavy in original: {len(heavy_df)}")
    print(f"-- len light in original: {len(light_df)}")

    prev_heavy = 0
    prev_light = 0

    len_heavy = len(heavy_df)
    len_light = len(light_df)

    out = {}
    stats = {}
    print(f"-- len of original: {len(df)}")
    for split_type, pct in split.items():

        boundary_heavy = round(pct * len_heavy)
        boundary_light = round(pct * len_light)

        split_heavy_df = heavy_df[prev_heavy: min(prev_heavy + boundary_heavy, len_heavy)]
        split_light_df = light_df[prev_light: min(prev_light + boundary_light, len_light)]

        out[f"{split_type}_{pct}"] = pd.concat([split_heavy_df, split_light_df])
        stats[f"{split_type}_{pct}/num_heavy_sequences"] = len(split_heavy_df)
        stats[f"{split_type}_{pct}/num_light_sequences"] = len(split_light_df)
        stats[f"{split_type}_{pct}/num_sequences"] = len(split_heavy_df) + len(split_light_df)

        prev_heavy = prev_heavy + boundary_heavy
        prev_light = prev_light + boundary_light

    return out, stats

def create_tfrecords(src_input_folder, out_tf_records_fldr, hash_partition, seed=1204):
    """
    src_input_folder: ex: /cb/ml/aarti/bayer_sample/paired/Eccles_2020/SRR10358525_paired
    """
    print(f"-- src_input_folder: {src_input_folder}")
    print(f"--- out_tf_records_fldr : {out_tf_records_fldr}")
    print(f"--- hash_partition : {hash_partition}")
    print(f"--- seed : {seed}")
    unit = get_sequence(oas_path=src_input_folder)

    src_input_folder = Path(src_input_folder)
    *_, oas_subset, study_id, unit_id = src_input_folder.parts

    
    df = unit.sequences_df()
    print(f"--- total_length BEFORE preprocessing: {len(df)}")


    if df.empty:
        sys.exit(f"--- No dataframe sequences in {src_input_folder}")

    dest_tfrecs_fldr = os.path.join(out_tf_records_fldr, oas_subset, study_id, unit_id)

    if not os.path.exists(dest_tfrecs_fldr):
        os.makedirs(dest_tfrecs_fldr)
    
    df = _preprocess_df(df, seed=seed)
    print(f"--- total_length AFTER preprocessing: {len(df)}")

    if hash_partition:
        partitions, stats_partitions = assign_mlsubset_by_sequence_hashing(df, seed=seed)
    else:
        split = {"train": 0.8, "val": 0.1, "test": 0.1}
        partitions, stats_partitions = partition_df(df, split, seed=seed)

    json_stats_dict = {}
    json_stats_dict.update(stats_partitions)

    for key, subset_df in partitions.items():

        # Shuffle again
        subset_df = subset_df.sample(frac=1, random_state=seed)

        out_tfrecord_name, out_stats_file_name = get_output_file_names(dest_tfrecs_fldr, unit_id, prefix=key)

        len_df = len(subset_df)
        num_tfrecs = int(len_df // 100000) + 1
            
        print(f"---dataframe rows : {key}: {len(subset_df)}")

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
        for tf_example, len_tokens in create_examples(subset_df, unit.study_year, unit.normalized_species, unit.oas_subset, unit.study_id, unit.unit_id):
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

        json_stats_dict[f"{key}_filename"] = out_tfrecord_name
        json_stats_dict[f"{key}_num_tf_examples"] = num_examples
        json_stats_dict[f"{key}_max_sequence_aa"] = max_length

    with open(out_stats_file_name, "w") as stats_fh:
        json.dump(json_stats_dict, stats_fh)



def main():
    """
    Main function
    """
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    create_tfrecords(args.src_input_folder, args.out_tf_records_fldr, int(args.hash_partition))


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()


    
    # create_tfrecords("/cb/ml/aarti/bayer_sample_new_datasets/unpaired/Banerjee_2017/SRR5060321_Heavy_Bulk", out_tf_records_fldr="/cb/ml/aarti/bayer_sample_filter_tfrecs")
    # create_tfrecords("/cb/customers/bayer/new_datasets/filters_default/unpaired/Li_2017/SRR3544217_Heavy_Bulk", out_tf_records_fldr="/cb/home/aarti/ws/code/bayer_tfrecs_filtering/tfrecord_scripts")

    # create_tfrecords("/cb/customers/bayer/new_datasets/filters_default/unpaired/Halliley_2015/SRR2088756_1_Heavy_IGHA", out_tf_records_fldr="/cb/home/aarti/ws/code/bayer_tfrecs_filtering/tfrecord_scripts", hash_partition=0)

    
    # create_tfrecords("/cb/customers/bayer/updated_dataset/filters_default_20211202/unpaired/Bender_2020/ERR3664761_Heavy_Bulk", out_tf_records_fldr="/cb/home/aarti/ws/code/bayer_tfrecs_filtering/tfrecord_scripts", hash_partition=0)
