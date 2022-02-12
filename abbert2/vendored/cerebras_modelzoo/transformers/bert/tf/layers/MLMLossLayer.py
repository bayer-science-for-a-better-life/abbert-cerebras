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

import tensorflow as tf
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.BaseLayer import BaseLayer


class MLMLossLayer(BaseLayer):
    """
    MLM loss layer

    :param bool boundary_casting: See documentation for ``BaseLayer``
    :param bool tf_summary: See documentation for ``BaseLayer``
    """

    def __init__(
        self, boundary_casting=False, tf_summary=False, **kwargs,
    ):
        super(MLMLossLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

    def call(self, masked_lm_ids, masked_lm_weights, logits, batch_size=None):
        """
        MLM loss. Based on
        https://github.com/google-research/bert/blob/master/run_pretraining.py

        :param Tensor masked_lm_ids: The target tokens for the masked
            positions of shape [batch_size, max_predictions_per_seq].
            Might be zero-padded (if the sequence is too short to
            have the maximum number of predictions).
        :param Tensor masked_lm_weights: The `label_weights` tensor
            of shape [batch_size, max_predictions_per_seq].
            Has a value of 0.0 for the padding predictions and arbitrary
            non-zero values for real predictions.
        :param Tensor logits: The logits tensor of shape
            [batch_size, max_predictions_per_seq, vocab_size].
        :param int batch_size: for scaling the loss
        :returns: The MLM loss scalar tensor.
        """
        batch_dim = logits.get_shape()[0]
        max_pred = logits.get_shape()[1]
        vocab_size = logits.get_shape()[2]
        logits = tf.reshape(logits, [batch_dim * max_pred, vocab_size])

        # log_softmax must be done in full precision
        log_probs = tf.cast(
            tf.nn.log_softmax(tf.cast(logits, tf.float32), axis=-1),
            logits.dtype,
        )

        label_ids = tf.reshape(masked_lm_ids, [-1])
        label_weights = tf.cast(
            tf.reshape(masked_lm_weights, [-1]), dtype=logits.dtype,
        )

        one_hot_labels = tf.one_hot(
            label_ids, depth=vocab_size, dtype=logits.dtype,
        )

        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1], name="per_example_loss",
        )
        masked_per_ex_loss = label_weights * per_example_loss

        loss = tf.reduce_sum(input_tensor=masked_per_ex_loss)
        return tf.cast(loss / batch_size, logits.dtype)
