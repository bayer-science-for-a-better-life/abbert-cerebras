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

try:
    from cerebras.tf.summary import cs1_enable_summaries as cs_enable_summaries
    from cerebras.tf.summary import cs1_disable_summaries as cs_disable_summaries
except ImportError:
    from contextlib import nullcontext as cs_enable_summaries
    from contextlib import nullcontext as cs_disable_summaries


def validate_host_call(host_call):
    if not host_call:
        return ()

    if not isinstance(host_call, (list, tuple)) or len(host_call) not in [2, 3]:
        raise ValueError("host_call must be an iterable with length 2 or 3.")

    if not callable(host_call[0]):
        raise ValueError("Expected first item of host_call to be a callable.")

    if not isinstance(host_call[1], (list, tuple)):
        raise ValueError("Expected second item of host_call to be an iterable.")

    if len(host_call) > 2 and not isinstance(host_call[2], (list, tuple)):
        raise ValueError("Expected third item of host_call to be an iterable.")

    return host_call


def host_call_to_eval_metric_ops(host_call):
    eval_metric_ops = host_call[0](*host_call[1])

    if isinstance(eval_metric_ops, dict):
        metric_ops_dict = eval_metric_ops
    elif isinstance(eval_metric_ops, (tuple, list)):
        metric_ops_dict = {
            f"elem_{index}": value
            for index, value in enumerate(eval_metric_ops)
        }
    else:
        raise ValueError("Invalid eval_metric_ops")

    new_eval_metric_ops = {}
    for key, value in metric_ops_dict.items():
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and tf.is_tensor(value[0])
        ):
            new_eval_metric_ops[key] = value

    return new_eval_metric_ops
