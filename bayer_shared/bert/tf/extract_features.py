import os
import sys
from typing import Sequence

import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import itertools
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from pathlib import Path
import json
from functools import partial

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from bayer_shared.bert.tf.model import model_fn
from bayer_shared.bert.tf.utils import get_params
from bayer_shared.bert.tf.run import create_arg_parser, run
from bayer_shared.bert.tf.utils import get_oas_vocab
from bayer_shared.bert.tf.input.Tokenization import BaseTokenizer


def to_parquet(df, path, compression='zstd', compression_level=20, **write_table_kwargs):
    path = Path(path)
    # noinspection PyArgumentList
    pq.write_table(pa.Table.from_pandas(df), path,
                   compression=compression, compression_level=compression_level,
                   **write_table_kwargs)


class ExtractEmbeddingsFromBert:

    def __init__(self, params, model_dir, checkpoint_path=None):
        self.params = params
        self.model_dir = model_dir
        self.args = args
        
        if self.params["runconfig"].get("mode") != "predict":
            tf.compat.v1.logging.info(f" Mode must be `predict` when extracting embeddings. Setting mode to `predict`")

            self.params["runconfig"]["mode"] = "predict"
        self.params["runconfig"]["validate_only"] = False
        self.params["runconfig"]["compile_only"] = False
        self.params["runconfig"]["checkpoint_path"] = checkpoint_path
        
        tf.compat.v1.logging.info(self.params)

        (
            self.vocab_word_to_id_dict,
            self.min_aa_id,
            self.max_aa_id,
        ) = get_oas_vocab(self.params["predict_input"].get("vocab_file"), self.params["predict_input"].get("dummy_vocab_size"))
        self.vocab_words = list(self.vocab_word_to_id_dict.keys())

        self.tokenizer = BaseTokenizer(self.vocab_word_to_id_dict)
        self.special_tokens = ["[CLS]", "[PAD]", "[MASK]", "[SEP]"]
        self.special_tokens_word_id_dict = self._get_special_token_ids(
            self.special_tokens, self.tokenizer
        )
        
        self.max_sequence_length = params["predict_input"]["max_sequence_length"]
        self.output_type_shapes = self.get_output_type_shapes(self.max_sequence_length)

    def _get_special_token_ids(self, special_tokens, tokenizer):
        """
        Function to get integer ids for special tokens.
        :param list special_tokens: special tokens in vocab.
        :param tokenizer: Tokenizer with function `check_word_and_get_id`
            which checks if the special token is part of vocab and returns it's id.
        :returns dict: dict with `special_tokens` as keys and their integer ids 
            as values.
        """
        special_tokens_word_id_dict = {}

        for key in special_tokens:
            special_tokens_word_id_dict[key] = tokenizer.check_word_and_get_id(
                key
            )

        return special_tokens_word_id_dict


    def build_predict_input_fn(self, data):

        def _generator_fn(data):

            for sequence in data:
                if isinstance(sequence, str):
                    sequence = list(sequence.strip())
                
                input_ids = [self.special_tokens_word_id_dict["[CLS]"]] + self.tokenizer.convert_tokens_to_ids(sequence) + [self.special_tokens_word_id_dict["[SEP]"]]

                # Input mask is  0's on non-padded & 1's on padded position 
                input_mask = [0] * len(input_ids)

                num_pad = self.max_sequence_length - len(input_ids)
                input_ids.extend([self.special_tokens_word_id_dict["[PAD]"]] * num_pad)
                input_mask.extend([1] * num_pad)

                assert len(input_ids) == self.max_sequence_length
                assert len(input_mask) == self.max_sequence_length

                # create feature dict
                features = dict()
                features["input_ids"] = input_ids
                features["input_mask"] = np.array(input_mask, dtype=np.int32)

                yield features, np.int32(np.empty(1)[0])

        # get actual tensorflow types
        types = (
            {
                k: getattr(tf, v["output_type"])
                for k, v in self.output_type_shapes.items()
            },
            tf.int32,
        )

        # get actual shapes
        shapes = (
            {k: v["shape"] for k, v in self.output_type_shapes.items()},
            [],
        )

        dataset = tf.data.Dataset.from_generator(
            partial(
                _generator_fn,
                data,
            ),
            output_types=types,
            output_shapes=shapes,
        )

        dataset = dataset.batch(self.params["predict_input"]["batch_size"], drop_remainder=True)

        return dataset


    def predict_input_fn(self, params, input_context=None):
        return self.build_predict_input_fn(self.data)

    
    def get_output_type_shapes(self, max_sequence_length):
        # process for output shapes and types
        output = {
            "input_ids": {"output_type": "int32", "shape": [max_sequence_length],},
            "input_mask": {"output_type": "int32", "shape": [max_sequence_length],},
        }

        return output
    
    def get_embeddings(self, data):

        self.data = data
        self.params["runconfig"]["predict_steps"] = len(data)

        model_output = run(
        args=self.args,
        params=self.params,
        model_fn=model_fn,
        train_input_fn=None,
        eval_input_fn=None,
        predict_input_fn=self.predict_input_fn,
        cs1_modes=["train", "eval", "predict"])

        vocab, _, _ = get_oas_vocab(self.params["predict_input"].get("vocab_file"), self.params["predict_input"].get("dummy_vocab_size"))

        if model_output:
            predict_output_dir = os.path.join(self.model_dir, "predict")
        
        if not os.path.exists(predict_output_dir):
            os.makedirs(predict_output_dir)

        vocab_file = os.path.join(self.model_dir, "predict", "vocab_file.json")
        with open(vocab_file, "w") as fh:
            json.dump(vocab, fh)

        file_content = []
        len_file_content = 0
        fileidx = 0
        parquet_file_path = os.path.join(predict_output_dir, f"extracted_features_{fileidx}.parquet")
        while True:
            try:
                if len_file_content == 100000:
                    df = pd.DataFrame(file_content)
                    print(df.head())
                    print(df.info())
                    to_parquet(df, parquet_file_path)
                    fileidx += 1
                    file_content = []
                    len_file_content = 0
                    parquet_file_path = os.path.join(predict_output_dir, f"extracted_features_{fileidx}.parquet")   
                else:
                    predictions = next(model_output)
                    tf.compat.v1.logging.info(f"prediction_output:{predictions}")
                    for key, val in predictions.items():
                        if key.startswith("encoder"):
                            predictions[key] = val.flatten().astype(np.float32)
                    file_content.append(predictions)
                    len_file_content += 1

            except StopIteration:
                df = pd.DataFrame(file_content)
                to_parquet(df, parquet_file_path)
                print(df.head())
                print(df.info())
                break

        pass



def main(params_file_path, model_dir, data, checkpoint_path=None):
    """
    Main function
    """
    # tf.compat.v1.disable_eager_execution()
    params = get_params(params_file_path)

    embed_obj = ExtractEmbeddingsFromBert(params, model_dir, checkpoint_path)

    # dataset = embed_obj.build_predict_input_fn(data)
    # it = tf.compat.v1.data.make_initializable_iterator(dataset)
    # next_batch = it.get_next()

    # with tf.compat.v1.Session() as sess:
    #     sess.run(tf.compat.v1.global_variables_initializer())
    #     sess.run(tf.compat.v1.tables_initializer())
    #     sess.run(it.initializer)

    #     for rep in range(10):
    #         if rep % 100 == 0:
    #             print(rep)
                
    #         data_batch = sess.run(next_batch)
    #         print(data_batch)

    embed_obj.get_embeddings(data)




if __name__ == "__main__":
    """
    Example usage
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where embeddings will be stored. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )

    args = parser.parse_args(sys.argv[1:])

    data = ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG","KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
    ]

    main(args.params, args.model_dir, data, args.checkpoint_path)
