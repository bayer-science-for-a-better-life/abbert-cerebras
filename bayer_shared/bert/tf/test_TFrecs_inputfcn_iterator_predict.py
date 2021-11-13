import os
import sys
import numpy as np

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from bayer_shared.bert.tf.input.Tokenization import BaseTokenizer
from bayer_shared.bert.tf.data import predict_input_fn as input_fn
from bayer_shared.bert.tf.utils import get_params
from bayer_shared.bert.tf.utils import get_oas_vocab
tf.compat.v1.disable_eager_execution()
import pandas as pd
import json

# number of batches to iterate over
NUM_BATCHES = 1

yaml_path = "/cb/home/aarti/ws/code/bayer_modelzoo/bayer_shared/bert/tf/configs/gpu_params_roberta_base_heavy_sequence_small_dataset_predict.yaml"
params = get_params(yaml_path)
print(params)

vocab_word_to_id_dict, min_aa_id, max_aa_id = get_oas_vocab(params["train_input"].get("dummy_vocab_size"))
vocab_words = list(vocab_word_to_id_dict.keys())

tokenizer = BaseTokenizer(vocab_word_to_id_dict)

dataset = input_fn(params)

it = tf.compat.v1.data.make_initializable_iterator(dataset)
next_batch = it.get_next()

# df = pd.read_parquet("/cb/home/aarti/ws/code/r09_no_debug_flow/modelzoo/transformers/bayer/tf/test_tfrecords_fldr/Setliff_2019_heavy_test.parquet")
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.tables_initializer())
    sess.run(it.initializer)

    for rep in range(NUM_BATCHES):
        if rep % 100 == 0:
            print(rep)
            
        data_batch = sess.run(next_batch)

        print(data_batch[0])
        print(tokenizer.convert_ids_to_tokens(data_batch[0]["input_ids"][0]))

    
        print("---------")


