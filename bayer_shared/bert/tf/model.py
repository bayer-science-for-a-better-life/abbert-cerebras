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

from bayer_shared.bert.tf.BayerBertModel import BayerBertModel
from modelzoo.common.estimator.tf.cs_estimator_spec import CSEstimatorSpec
from modelzoo.common.hooks.grad_accum_hooks import get_grad_accum_hooks


def model_fn(features, labels, mode, params):
    """
    The model function to be used with TF estimator API.
    """
    encode_only = mode == tf.estimator.ModeKeys.PREDICT
    bert = BayerBertModel(params, encode_only)
    outputs = bert(features, mode)
    

    total_loss = None
    host_call = None
    predictions = None
    train_op = None

    if mode != tf.estimator.ModeKeys.PREDICT:
        total_loss = bert.build_total_loss(outputs, features, labels, mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = bert.build_train_ops(total_loss)
    elif mode == tf.estimator.ModeKeys.EVAL:
        # Note that we are doing argmax on the fabric, not on the host.
        # This is somewhat inconsistent with metrics being processed on the host,
        # but is done primarily to improve performance.
        # build inputs to eval metrics on the fabric to minimize work done on the host.
        eval_metric_inputs = bert.build_eval_metric_inputs(
            outputs, labels, features
        )
        host_call = (
            bert.build_eval_metric_ops,
            [eval_metric_inputs, labels, features],
        )
        train_op = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
        _, _, encoder_outputs = outputs
        predictions = {}
        predictions["features/input_ids"] = features["input_ids"]
        predictions["features/input_mask"] = features["input_mask"]
        predictions["features/unit_id"] = features["unit_id"]
        predictions["features/study_id"] = features["study_id"]
        predictions["features/study_year"] = features["study_year"]
        predictions["features/oas_subset"] = features["oas_subset"]
        if isinstance(encoder_outputs, list):
            for idx, value in enumerate(encoder_outputs):
                tf.compat.v1.logging.debug(f"---encoder_shape: {value}")
                predictions[f"encoder_layer_{idx}_output"] = value
        else:
            tf.compat.v1.logging.debug(f"---non list encoder_shape: {encoder_outputs}")
            predictions[f"encoder_layer_{params['model']['num_hidden_layers']-1}_output"] = encoder_outputs
        
    else:
        raise ValueError(f"Mode {mode} not supported.")

    hooks = None
    if bert.trainer.is_grad_accum():
        hooks = get_grad_accum_hooks(
            bert.trainer,
            runconfig_params=params["runconfig"],
            summary_dict={
                "train/cost": total_loss,
                "train/cost_masked_lm": bert.mlm_loss,
            },
            logging_dict={"loss": total_loss},
        )

    espec = CSEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        training_hooks=hooks,
        host_call=host_call,
        predictions=predictions
    )

    return espec
