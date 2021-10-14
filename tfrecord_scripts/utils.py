import tensorflow as tf
import numpy as np



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



