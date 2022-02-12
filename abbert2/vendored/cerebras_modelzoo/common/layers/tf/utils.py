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


def summarize_tensor(x, is_gradient_tensor=False, op_name_if_grad=None):
    """Attach a summary to the specified tensor.

    This function is added separately from the ``summary_layer`` below,
    because some tensors might be generated using the ``tf.custom_gradients``,
    which would disallow finding the gradient tensor using ``summary_layer``.
    In such case, the user writing a custom gradient must manually specify
    the gradient tensor to summarize.

    Args:
        x (tf.Tensor): Tensor to summarize.
        is_gradient_tensor (bool): Boolean that decides whether to designate
            the tensor's name as within the gradients.
        op_name_if_grad (str): If ``is_gradient_tensor==True``, then use
            this name, prefixed by ``"gradients/"``, as the summary name.
    """

    display_name = x.op.name
    op_name = display_name
    # confusing because tensorboard shows display_name, and graph shows name
    # but when we extract to np format, we map to display name,
    # so display name need to be the op name.
    if is_gradient_tensor:
        if op_name_if_grad is None:
            # Don't have the original op name, so need to just use the grad
            # tensor's op name
            op_name_if_grad = op_name
        # Store both our "custom" name and the "real" op name
        op_name = f"gradients/{op_name_if_grad}"
    tf.compat.v1.summary.tensor_summary(
        name=op_name, display_name=display_name, tensor=x
    )


@tf.custom_gradient
def summary_layer(x):
    """Attaches a summary to a tensor and its delta.

    Args:
        x (tf.Tensor): Tensor to summarize.
    """

    summarize_tensor(x)

    def grad(dy):
        summarize_tensor(dy, is_gradient_tensor=True, op_name_if_grad=x.op.name)
        return dy

    return x, grad


def boundary_cast(x, name=None):
    """Internal measure to ensure matching between Tensorflow and the CS-1.

    Closest operation to match the mixed precision on the CS-1.

    Args:
        x: The tensor to boundary cast.

    Returns:
        x: The tensor ``x`` cast down to ``float16`` and back to ``float32``.
    """

    x = tf.cast(x, tf.float16, name=name)
    x = tf.cast(x, tf.float32, name=name)
    return x
