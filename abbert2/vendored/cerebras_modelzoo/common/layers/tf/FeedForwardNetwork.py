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

from abbert2.vendored.cerebras_modelzoo.common.layers.tf.ActivationLayer import ActivationLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.BaseLayer import BaseLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.DenseLayer import DenseLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.DropoutLayer import DropoutLayer


class FeedForwardNetwork(BaseLayer):
    """A feed forward network that consists of a stack of fully connected\
    layers.

    Args:
        layers_units (int): List of units for each layer.
        layers_activation (str): List of activation types (str) for each layer.
        layers_dropout_rates (float): List of dropout rates (float) for each
            layer.
        use_bias (bool): If ``True``, use bias throughout all layers.
        kernel_initializer: Kernel initializer. Defaults to
            ``"glorot_uniform"``.
        bias_initializer: Bias initializer. Defaults to ``"zeros"``.
        kernel_regularizer (callable): Kernel regularizer.
        bias_initializer (callable): Bias regularizer.

    """

    def __init__(
        self,
        layers_units,
        layers_activation=None,
        layers_dropout_rates=None,
        use_bias=False,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        """Initialize the FFN object instance.
        """

        super(FeedForwardNetwork, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.num_dense_layers = len(layers_units)
        assert (
            self.num_dense_layers > 0
        ), "Number of dense layers should be at least 1."

        if layers_activation is not None:
            assert len(layers_activation) == self.num_dense_layers, (
                "len(layers_activation) should equal the number"
                " of dense layers."
            )
        if layers_dropout_rates is not None:
            assert len(layers_dropout_rates) == self.num_dense_layers, (
                "len(layers_dropout) should equal the number" "of dense layers."
            )

        self.layers = []
        for dense_layer in range(self.num_dense_layers):
            self.layers.append(
                DenseLayer(
                    layers_units[dense_layer],
                    None,
                    use_bias,
                    kernel_initializer,
                    bias_initializer,
                    kernel_regularizer,
                    bias_regularizer,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

            # Activation
            self.layers.append(
                ActivationLayer(
                    activation=layers_activation[dense_layer]
                    if layers_activation is not None
                    else None,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

            # Dropout
            self.layers.append(
                DropoutLayer(
                    rate=layers_dropout_rates[dense_layer]
                    if layers_dropout_rates is not None
                    else 0.0,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.dtype_policy,
                )
            )

    def call(self, inputs, training=True, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return inputs
