import tensorflow as tf
import sys

from bayer_shared.bert.tf.input.OasMlmOnlyTfRecordsDynamicMaskProcessor import (
    OasMlmOnlyTfRecordsDynamicMaskProcessor,
)

def train_input_fn(params, input_context=None):
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_tf_dataset(
        mode=tf.estimator.ModeKeys.TRAIN, input_context=input_context
    )


def eval_input_fn(params, input_context=None):
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_tf_dataset(
        mode=tf.estimator.ModeKeys.EVAL, input_context=input_context
    )

