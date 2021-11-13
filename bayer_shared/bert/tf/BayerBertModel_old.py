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
from modelzoo.transformers.bert.tf.BertModel import BertModel
from modelzoo.common.metrics.utils import (
    aggregate_across_replicas,
    streaming_confusion_matrix,
)


class BayerBertModel(BertModel):
    """
    The Genentech BERT model https://arxiv.org/abs/1906.08230.
    Differs from Bert model only in eval metrics computations.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params, encode_only=False):

        super(BayerBertModel, self).__init__(
            params=params, encode_only=encode_only
        )
        self.encode_only = encode_only

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        metrics_dict = super().build_eval_metric_ops(eval_metric_inputs, labels, features)

        nsp_output, mlm_pred, mlm_xentr = eval_metric_inputs
        labels = features["masked_lm_ids"]
        weights = tf.where(features["masked_lm_weights"] > 0, 1, 0)
        num_classes = self.vocab_size

        metrics_dict["confusion_matrix"] = streaming_confusion_matrix(labels, mlm_pred, num_classes, weights)

        return metrics_dict

    

