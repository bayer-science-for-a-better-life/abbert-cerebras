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
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.EmbeddingLayer import EmbeddingLayer
from abbert2.vendored.cerebras_modelzoo.common.layers.tf.PositionEmbeddingLayer import (
    PositionEmbeddingLayer,
)


def create_embedding_layers(
    vocab_size,
    embedding_size,
    segment_embedding_size=None,
    embeddings_initializer='uniform',
    bias_initializer='zeros',
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    use_bias=False,
    max_position_embeddings=None,
    position_embeddings_type=None,
    position_embeddings_initializer='uniform',
    position_embeddings_regularizer=None,
    num_segments=None,
    segment_embeddings_initializer='uniform',
    segment_embeddings_regularizer=None,
    boundary_casting=False,
    tf_summary=False,
    dtype=None,
):
    """
    Creates token and, optionally,position and segment embeddings.

    :param int vocab_size: Size of input vocabulary.
    :param int embedding_size: Dimension of the embedding space.
    :param int segment_embedding_size: Dimension of the embedding space for segment
        embeddings. Useful when factorized embeddings are used for tokens and
        so the size of the embedding space for segments differs from that for
        tokens. Defaults to the same value as embedding_size.
    :param Optional[str,Callable] embeddings_initializer: Token embeddings
        initializer. Defaults to 'uniform'.
    :param Optional[string,Callable] bias_initializer: Token embeddings
        bias initializer. Defaults to 'zeros'.
    :param Optional[Callable] embeddings_regularizer: Tokens
        embeddings regularizer. Defaults to None.
    :param Optional[Callable] activity_regularizer: Token embeddings
        activation regularizer. Defaults to None.
    :param Optional embeddings_constraint: Token ebeddings constraint.
        Defaults to None.
    :param Optional[bool] mask_zero: Whether or not the input value 0 is a
        special "padding" value that should be masked out. Defaults to False.
    :param Optional[bool] use_bias: Whether to use bias for token embeddings.
        Defaults to False.
    :param Optional[int] max_position_embeddings: Maximum sequence length to train
        using model. If None (default), set to input sequence length.
    :param str position_embeddings_type: 'learned' or 'fixed'. Defaults to None,
        in which case position embeddings are not created.
    :param Optional[str,Callable] position_embeddings_initializer: Position
        embeddings initializer. Defaults to "uniform".
    :param Optional[Callable] position_embeddings_regularizer: Position
        embeddings regularizer. Defaults to None.
    :param Optional[int] num_segments: Number of segments for the segment
        embedding layer. Defaults to None, in which case the segment embedding
        layer is not created.
    :param Optional[str,Callable] segment_embeddings_initializer: Segment
        embeddings initializer. Defaults to "uniform".
    :param Optional[Callable] segment_embeddings_regularizer: Segment
        embeddings regularizer. Defaults to None.
    :param bool boundary_casting: Flag to enable boundary casting.
        Defaults to False.
    :param tf_summary: Flag to enable debug summaries. Defaults to False.
    :param dtype: Dtype or policy. Defaults to None.

    Returns:
       Token, position, and embedding layers
    """

    if segment_embedding_size is None:
        segment_embedding_size = embedding_size

    token_embedding = EmbeddingLayer(
        input_dim=vocab_size,
        output_dim=embedding_size,
        embeddings_initializer=embeddings_initializer,
        embeddings_regularizer=embeddings_regularizer,
        boundary_casting=boundary_casting,
        tf_summary=tf_summary,
        dtype=dtype,
        name="input_embedding",
    )

    position_embedding = (
        PositionEmbeddingLayer(
            max_position_embeddings=max_position_embeddings,
            embedding_type=position_embeddings_type,
            embeddings_initializer=position_embeddings_initializer,
            embeddings_regularizer=position_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="position_embedding",
        )
        if position_embeddings_type
        else None
    )

    segment_embedding = (
        EmbeddingLayer(
            input_dim=num_segments,
            output_dim=segment_embedding_size,
            embeddings_initializer=segment_embeddings_initializer,
            embeddings_regularizer=segment_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="segment_embedding",
            weight_name="segment_embedding_weights",
        )
        if num_segments
        else None
    )

    return token_embedding, position_embedding, segment_embedding


def create_autoregressive_attention_mask(
    max_sequence_length, batch_size=1, dtype=None
):
    """
    Create autoregressive (triangular) mask.

    :param int batch_size: Batch size.
    :param int max_sequence_length: Max sequence length.
    :param dtype: Dtype of the resulting mask.

    Returns:
        The autoregressive mask of shape
        [batch_size, max_sequence_length, max_sequence_length].
    """

    # Triangular mask
    with tf.compat.v1.variable_scope('autoregressive_mask'):
        # The first dimension here is the query sequence length, and the
        # second dimension is the key sequence length. An autoregressive
        # model permits each query to attend to all keys up to and
        # including the position of the query, so each row, `i`, should
        # mask all positions after position `i`.
        diag_vals = tf.ones(
            [max_sequence_length, max_sequence_length], dtype=dtype
        )
        # The tril looks like:
        # [ 1, 0, 0,
        #   1, 1, 0,
        #   1, 1, 1 ]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        # Swap 0s and 1s since we use 1 to indicate masked positions
        tril = 1 - tril
        # Expand the batch dimension
        auto_attn_mask = tf.tile(
            tf.expand_dims(tril, axis=0), [batch_size, 1, 1]
        )

    return auto_attn_mask
