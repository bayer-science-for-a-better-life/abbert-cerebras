import numpy as tf

import os
from glob import glob
import tensorflow as tf
from pathlib import Path

tf.compat.v1.enable_eager_execution()


data_dir = ["/cb/home/aarti/ws/code/bayer_tfrecs_filtering/tfrecord_scripts/unpaired/Bender_2020/ERR3664761_Heavy_Bulk/*train_80*.tfrecord"]
input_files = []
for folder in data_dir:
    parent_folder, glob_pattern = os.path.split(folder)
    files_matched = list(Path(parent_folder).rglob(glob_pattern))
    files_str = [str(element) for element in files_matched]
    input_files.extend(files_str)

print(input_files)
raw_dataset = tf.data.TFRecordDataset(input_files)


def _parse_raw_tfrecord(raw_record):
    feature_map = {"tokens": tf.io.FixedLenSequenceFeature((), tf.string)}
    context_features_map = {}

    ## Note: Uncomment the following lines only if these fields are needed for some processing       
    context_features_map["study_year"] = tf.io.FixedLenFeature([], tf.int64)
    context_features_map["oas_subset"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["species"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["unit_id"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["study_id"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["index_in_unit"] = tf.io.FixedLenFeature([], tf.int64)
    context_features_map["chain"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["v_call"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["j_call"] = tf.io.FixedLenFeature([], tf.string)
    context_features_map["subject"] = tf.io.FixedLenFeature([], tf.string)

    context_features_map["cdr_start"] = tf.io.FixedLenFeature([3], tf.int64)
    context_features_map["cdr_length"] = tf.io.FixedLenFeature([3], tf.int64)
    context_features_map["fw_start"] = tf.io.FixedLenFeature([4], tf.int64)
    context_features_map["fw_length"] = tf.io.FixedLenFeature([4], tf.int64)

    context_features, raw_features = tf.io.parse_single_sequence_example(
        raw_record, sequence_features=feature_map, context_features=context_features_map,
    )

    features = {
        "tokens": raw_features["tokens"],
        "cdr_start": context_features["cdr_start"],
        "cdr_length": context_features["cdr_length"],
        "fw_start": context_features["fw_start"],
        "fw_length": context_features["fw_length"],

        ## Note: Uncomment the following lines only if these fields are needed for some processing  
        "study_year": context_features["study_year"],
        "oas_subset": context_features["oas_subset"],
        "species": context_features["species"],
        "unit_id": context_features["unit_id"],
        "study_id": context_features["study_id"],
        "index_in_unit": context_features["index_in_unit"],
        "chain": context_features["chain"],
        "v_call": context_features["v_call"],
        "j_call": context_features["j_call"],
        "subject": context_features["subject"]

    }

    return features


parsed_dataset = raw_dataset.map(_parse_raw_tfrecord)
n = 0
for parsed_record in parsed_dataset:
    print(repr(parsed_record))
    n +=1
    print("-------")

print(f"num_examples:{n}")