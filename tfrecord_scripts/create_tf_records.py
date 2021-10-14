from abbert2.oas.oas import train_validation_test_iterator
from abbert2.oas.oas import sapiens_like_train_val_test
from tfrecord_scripts.utils import (
    create_bytes_feature,
    create_int_feature,
)

from pathlib import Path
import os
import tensorflow as tf

import json
import argparse
import sys

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

    # Drop rows with negative values in index and length columns
    query = ""
    for col in df.columns:
        if "aligned_sequence" not in col:
            query = query + f"{col} >= 0 and "
    query = query[:-5]  # To remove final ` and `

    df = df.query(query)

    return df


def get_iterator(oas_path):

    def partitioner(oas_path):
        return lambda : sapiens_like_train_val_test(oas_path=oas_path)

    iterator = train_validation_test_iterator(partitioner=partitioner(oas_path))

    return iterator


def create_examples(df, chain, unit):

    feature = {}
    study_year = unit.study_year
    normalized_species = unit.normalized_species
    oas_subset, study_id, unit_id = unit.id

    for _, row in df.iterrows():
        try:
            feature["tokens"] = row[f"aligned_sequence_{chain}"]
            feature["cdr_start"] = [int(row[f"cdr1_start_{chain}"]), int(row[f"cdr2_start_{chain}"]), int(row[f"cdr3_start_{chain}"])]
            feature["cdr_length"] = [int(row[f"cdr1_length_{chain}"]), int(row[f"cdr2_length_{chain}"]), int(row[f"cdr3_length_{chain}"])]
            feature["fw_start"] = [int(row[f"fw1_start_{chain}"]), int(row[f"fw2_start_{chain}"]), int(row[f"fw3_start_{chain}"]), int(row[f"fw4_start_{chain}"])]
            feature["fw_length"] = [int(row[f"fw1_length_{chain}"]), int(row[f"fw2_length_{chain}"]), int(row[f"fw3_length_{chain}"]), int(row[f"fw4_length_{chain}"])]

            
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
    
    context_features_dict["cdr_start"] = create_int_feature(feature["cdr_start"])
    context_features_dict["cdr_length"] = create_int_feature(feature["cdr_length"])
    context_features_dict["fw_start"] = create_int_feature(feature["fw_start"])
    context_features_dict["fw_length"] = create_int_feature(feature["fw_length"])

    context = tf.train.Features(feature=context_features_dict)
    
    tf_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
    return tf_example


def get_output_file_names(dest_tfrecs_fldr, unit_id, chain, ml_subset):

    ml_subset_dict = {
        "train": "lte2017",
        "validation": "eq2018",
        "test": "gte2019"
    }

    out_tfrecord_name = f"{chain}_{ml_subset_dict[ml_subset]}_{unit_id}.tfrecord"
    out_tfrecord_name = os.path.join(dest_tfrecs_fldr, out_tfrecord_name)

    out_stats_file_name = f"{chain}_{ml_subset_dict[ml_subset]}_{unit_id}_stats.json"
    out_stats_file_name = os.path.join(dest_tfrecs_fldr, out_stats_file_name)

    return out_tfrecord_name, out_stats_file_name


def create_tfrecords(src_input_folder, out_tf_records_fldr):
    """
    src_input_folder: ex: /cb/ml/aarti/bayer_sample/paired/Eccles_2020/SRR10358525_paired
    """
    print(f"-- src_input_folder={src_input_folder}")
    print(f"--- out_tf_records_fldr : {out_tf_records_fldr}")
    iterator = get_iterator(oas_path=src_input_folder)

    src_input_folder = Path(src_input_folder)
    *_, oas_subset, study_id, unit_id = src_input_folder.parts

    dest_tfrecs_fldr = os.path.join(out_tf_records_fldr, oas_subset, study_id, unit_id)

    if not os.path.exists(dest_tfrecs_fldr):
        os.makedirs(dest_tfrecs_fldr)

    for unit, chain, ml_subset, df in iterator:
        
        num_examples = 0
        max_length = float("-inf")
        out_tfrecord_name, out_stats_file_name = get_output_file_names(dest_tfrecs_fldr, unit_id, chain, ml_subset)

        if all([chain, ml_subset, not df.empty]):
            #write to tf
            df = _preprocess_df(df)
        
            if df.empty:
                print(f"--- dataframe empty after preprocess")
                continue

            print(f"---dataframe rows after preprocess : {len(df)}, {chain}, {ml_subset}")

            writer = tf.io.TFRecordWriter(out_tfrecord_name)
            for tf_example, len_tokens in create_examples(df, chain, unit):

                if tf_example is not None and len_tokens is not None:
                    writer.write(tf_example.SerializeToString())
                    num_examples += 1
                    max_length = max(max_length, len_tokens)

                if num_examples % 5000 == 0:
                    print(f"---{out_tfrecord_name} -  Wrote {num_examples} examples so far ...")
            
            print(f"----DONE: {out_tfrecord_name} -  Wrote {num_examples} examples")
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
                    "num_examples": "file_corrupted", 
                    "max_aligned_sequence_length": "file_corrupted"
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
