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



import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import io
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet as pq
from pathlib import Path
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from bayer_shared.bert.tf.data import eval_input_fn, train_input_fn, predict_input_fn
from bayer_shared.bert.tf.model import model_fn
from bayer_shared.bert.tf.utils import get_params
from bayer_shared.bert.tf.run import create_arg_parser, run
from bayer_shared.bert.tf.utils import get_oas_vocab

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(30, 30))
    cm = cm.astype(int)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm_tf = tf.math.divide_no_nan(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
    cm = cm_tf.numpy()
    cm = np.round(cm, decimals=2)
    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image


def to_parquet(df, path, compression='zstd', compression_level=20, **write_table_kwargs):
    path = Path(path)
    # noinspection PyArgumentList
    pq.write_table(pa.Table.from_pandas(df), path,
                   compression=compression, compression_level=compression_level,
                   **write_table_kwargs)


def main():
    """
    Main function
    """
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    out = run(
        args=args,
        params=params,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        predict_input_fn=predict_input_fn,
        cs1_modes=["train", "eval"],
    )
    vocab, _, _ = get_oas_vocab(params["predict_input"].get("dummy_vocab_size"))

    if out and args.mode == "eval":
        fname = os.path.join(args.model_dir, "eval", f"eval_{out['global_step']}.npy")
        np.save(fname, out)

        file_writer_cm = tf.summary.create_file_writer(os.path.join(args.model_dir, "eval", "cm"))
        figure = plot_confusion_matrix(out["eval/confusion_matrix"], class_names=list(vocab.keys()))
        cm_image = plot_to_image(figure)
    
        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=out["global_step"])

    if out and args.mode == "predict":
        predict_output_dir = os.path.join(args.model_dir, "predict")
        
        if not os.path.exists(predict_output_dir):
            os.makedirs(predict_output_dir)

        vocab_file = os.path.join(args.model_dir, "predict", "vocab_file.json")
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
                    predictions = next(out)
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
                

if __name__ == "__main__":
    """
    Example usage
    ~/miniconda3/envs/bayer/bin/python run_eval_predict.py --mode=predict --params=/cb/home/aarti/ws/code/bayer_for_santi/modelzoo/transformers/bayer/tf/configs/feature_extract/cs1_params_roberta_base_heavy_sequence_LRSwarm_decay_bsz1k_msl164_cdr_25_25_50.yaml --checkpoint_path=/cb/ml/aarti/bayer_convergence/r09_sysf23/cs1_params_roberta_base_heavy_sequence_LRSwarm_decay_bsz1k_msl164_cdr_25_25_50/model.ckpt-540000 --model_dir=<path/to/folder/where/extracted_parquets_to_be_stored>
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
