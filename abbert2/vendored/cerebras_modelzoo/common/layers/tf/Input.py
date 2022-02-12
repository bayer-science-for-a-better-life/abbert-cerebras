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

from abbert2.vendored.cerebras_modelzoo.common.layers.tf.utils import summary_layer


def SetupInputTensor(features, tf_summary=False):
    """Adds tensor summary to the model's input features and their gradient, if
    ``tf_summary`` is set to ``True``.

    Args:
        features: The input features.
    """

    output = features
    if tf_summary:
        output = summary_layer(output)
    return output
