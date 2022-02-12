# Copyright 2021 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Function for performing standard transformations on datasets.
"""
import os
import shutil

import numpy as np
import tensorflow as tf


def transform_dataset(
    dataset,
    map_fn,
    batch_size,
    is_training,
    shuffle,
    post_batch_map_fn=None,
    shuffle_buffer=None,
    repeat=True,
    seed=None,
    map_before_batch=False,
    batch_fn=None,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
):
    """
    Apply standard transformations to a dataset:
        - shuffle -> batch -> map -> repeat if map_before_batch is False
        - shuffle -> map -> batch -> repeat if map_before_batch is True

    Batching before mapping is generally faster and the preferred method due to
    vectorization of map fn.

    Note: Mapping before batching may be required if parsing TF records that
    contain `FixedLenSequenceFeature` examples (rather than `FixedLenFeature`)

    :param tf.data.Dataset dataset: Dataset to apply transformations to
    :param func map_fn: Mapping function to be applied after batching data
    :param int batch_size: Batch size for model training
    :param bool shuffle: If True, then shuffle the dataset
    :param int shuffle_buffer: Size of shuffle buffer to sample data from
    :param bool repeat: If True, repeat the dataset
    :param int seed: Seed to use for shuffle randomizer or None
    :param bool map_before_batch: if True, mapping will happen before batching.
    :param tf.Tensor num_parallel_calls: representing the number of batches to compute
           asynchronously in parallel. Default value is `tf.data.experimental.AUTOTUNE` when
           number of parallel calls is set dynamically based on available resources.
    :returns: tf dataset
    """

    if batch_fn is None:
        batch_fn = lambda ds: ds.batch(batch_size, drop_remainder=True)

    if is_training and shuffle:
        if not shuffle_buffer:
            shuffle_buffer = 10 * batch_size
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    if not map_before_batch:
        dataset = batch_fn(dataset)

    if map_fn is not None:
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=num_parallel_calls,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (shuffle and seed is None),
        )

    if map_before_batch:
        dataset = batch_fn(dataset)

    if post_batch_map_fn:
        dataset = dataset.map(
            post_batch_map_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=not (shuffle and seed is None),
        )

    if is_training and repeat:
        dataset = dataset.repeat()

    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def check_and_create_output_dirs(output_dir, filetype="tfrecord"):
    does_exist = False
    if os.path.isdir(output_dir):
        for fname in os.listdir(output_dir):
            if filetype in fname:
                does_exist = True
                break

    if does_exist:
        _in = input(
            "Output directory of same name already contains tfrecords."
            + " Do you want to delete the folder to write"
            + " new records in the same output folder name? (yes/no): "
        )
        if _in.lower() in ["y", "yes"]:
            shutil.rmtree(output_dir)
        elif _in.lower() in ["n", "no"]:
            raise IsADirectoryError(
                "Create a new folder for the tfrecords you want to write!!"
            )
        else:
            raise ValueError(f"Inputs can be yes, no, y or n. Received {_in}!!")

    try:
        os.makedirs(output_dir)
    except OSError as e:
        raise ValueError(
            f"Path {output_dir} is invalid. Validate the input arguments!!"
        )


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_int_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    if values is None:
        values = []
    if isinstance(values, np.ndarray) and values.ndim > 1:
        values = values.reshape(-1)
    if not isinstance(values, list):
        values = values.tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def create_float_feature(values):
    """Returns a float_list from a float / double."""
    if values is None:
        values = []
    if isinstance(values, np.ndarray) and values.ndim > 1:
        values = values.reshape(-1)
    if not isinstance(values, list):
        values = values.tolist()
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))
