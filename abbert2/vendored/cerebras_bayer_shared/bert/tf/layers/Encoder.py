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

from abbert2.vendored.cerebras_modelzoo.common.layers.tf.AddLayer import AddLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.AttentionLayer import SelfAttentionLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.BaseLayer import BaseLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.DropoutLayer import DropoutLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.FeedForwardNetwork import FeedForwardNetwork
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.PrePostProcessWrapper import (
    PrePostProcessWrapper,
)


class Encoder(BaseLayer):
    """
    Transformer encoder.
    Made up of num_hidden_layers identical layers.
    Each layer is composed of the sublayers:
        1. Self-attention layer
        2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        num_hidden_layers,
        filter_size,
        use_projection_bias_in_attention=False,
        use_ffn_bias_in_attention=False,
        use_ffn_bias=False,
        attention_initializer="glorot_uniform",
        ffn_initializer="glorot_uniform",
        weight_regularizer=None,
        attention_type="scaled_dot_product",
        attention_dropout_rate=0.0,
        nonlinearity="gelu",
        dropout_rate=0.0,
        dropout_seed=None,
        disable_layer_norm=False,
        layer_norm_epsilon=1e-8,
        use_pre_normalization=False,
        boundary_casting=False,
        tf_summary=False,
        all_encoder_outputs=False,
        **kwargs,
    ):
        """
        Initialize the encoder object instance.

        :param int hidden_size: Hidden size
        :param int num_heads: Number of heads in the multi-headed attention layer
        :param int num_hidden_layers: Number of encoders in the encoder stack
        :param int filter_size: Output size of the first layer of the FFN sublayer
        :param bool use_projection_bias_in_attention: Add bias to Q,K,V projections
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias_in_attention: Add bias in the concluding FFN
            in the Attention layer. Defaults to False.
        :param bool use_ffn_bias: Add bias in all dense layers of the encoder's ffn sublayer
        :param string initializer: Kernel initializer. Defaults to "glorot_uniform".
        :param callable weight_regularizer: Weights regularizer.
        :param string attention_type: Type of attention.
            Currently support {"dot_product","scaled_dot_product"}.
            Defaults to "scaled_dot_product".
        :param float attention_dropout_rate: Attention dropout rate.
        :param string nonlinearity: Type of nonlinearity applied after the first dense layer
            of the FFN sublayer. Defaults to gelu.
        :param float dropout_rate: Dropout rate. Same for all dense layers
            encountered in the Encoder. Defaults to 0 (i.e., no dropout)
        :param int dropout_seed: seed for initializing dropout layers.
        :param bool disable_layer_norm: Layer norm disabled if set to True.
        :param float layer_norm_epsilon: LayerNorm epsilon. Same for all layers
            encountered in the Encoder. Defaults to 1e-8
        :param bool use_pre_normalization: If False, uses original transformer encoder:
            "residual split -> transform -> dropout -> residual add -> layer norm".
            Otherwise, use residual block with pre-normaliztion:
            "residual split -> layer norm -> transform -> dropout -> residual add".
            Here transform is either a self-attention or ffn sub-block.
            Defaults to False.
        """
        super(Encoder, self).__init__(boundary_casting, tf_summary, **kwargs)

        self.all_encoder_outputs = all_encoder_outputs

        # Set up pre- and post-processing steps to be wrapped
        # around all encoder sublayers in the encoder stack
        pre_process_config = []
        if use_pre_normalization and not disable_layer_norm:
            pre_process_config.append(
                (
                    LayerNormalizationLayer,
                    {
                        'epsilon': layer_norm_epsilon,
                        'beta_regularizer': weight_regularizer,
                        'gamma_regularizer': weight_regularizer,
                        'boundary_casting': boundary_casting,
                        'tf_summary': tf_summary,
                        'dtype': self.dtype_policy,
                    },
                )
            )

        post_process_config = [
            (
                DropoutLayer,
                {
                    'rate': dropout_rate,
                    'seed': dropout_seed,
                    'boundary_casting': boundary_casting,
                    'tf_summary': tf_summary,
                    'dtype': self.dtype_policy,
                },
            ),
            (
                AddLayer,
                {
                    'boundary_casting': boundary_casting,
                    'tf_summary': tf_summary,
                    'dtype': self.dtype_policy,
                },
            ),
        ]

        if not use_pre_normalization and not disable_layer_norm:
            post_process_config.append(
                (
                    LayerNormalizationLayer,
                    {
                        'epsilon': layer_norm_epsilon,
                        'beta_regularizer': weight_regularizer,
                        'gamma_regularizer': weight_regularizer,
                        'boundary_casting': boundary_casting,
                        'tf_summary': tf_summary,
                        'dtype': self.dtype_policy,
                    },
                )
            )

        # Create sublayers for each layer.
        self.layers = []
        for _ in range(num_hidden_layers):
            self_attention_layer = SelfAttentionLayer(
                hidden_size,
                num_heads,
                use_projection_bias=use_projection_bias_in_attention,
                use_ffn_bias=use_ffn_bias_in_attention,
                initializer=attention_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                attention_type=attention_type,
                dropout_rate=attention_dropout_rate,
                dropout_seed=dropout_seed,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="self_attention",
            )

            feed_forward_network = FeedForwardNetwork(
                layers_units=[filter_size, hidden_size],
                layers_activation=[nonlinearity, None],
                layers_dropout_rates=[0.0, 0.0],
                use_bias=use_ffn_bias,
                kernel_initializer=ffn_initializer,
                kernel_regularizer=weight_regularizer,
                bias_regularizer=weight_regularizer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.dtype_policy,
                name="ffn",
            )

            self.layers.append(
                [
                    PrePostProcessWrapper(
                        self_attention_layer,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                    PrePostProcessWrapper(
                        feed_forward_network,
                        pre_process_config,
                        post_process_config,
                        dtype=self.dtype_policy,
                    ),
                ]
            )

    def call(self, inputs, self_attention_mask=None, training=True):
        encoder_all_layers_outputs = []
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]
            inputs = self_attention_layer(
                inputs, mask=self_attention_mask, training=training
            )
            inputs = feed_forward_network(inputs, training=training)
            if self.all_encoder_outputs:
                # inputs would be shape (bsz, msl, hidden_size)
                encoder_all_layers_outputs.append(inputs)

        return inputs, encoder_all_layers_outputs
