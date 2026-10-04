"""Explicit additive attention over sequence hidden states."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class PaddingMask(layers.Layer):
    """Build a float mask where non-zero token ids are 1.0 (serializable)."""

    def call(self, inputs):
        return tf.cast(tf.not_equal(inputs, 0), tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


class AdditiveAttention(layers.Layer):
    """Bahdanau-style attention implemented explicitly for interpretability.

    Given BRNN outputs H = [h_1, ..., h_T]:
      score_i = v^T tanh(W h_i)
      alpha_i = softmax(score_i)
      context = sum_i alpha_i h_i
    """

    def __init__(self, units: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = None
        self.v = None

    def build(self, input_shape):
        # When called as layer(seq, mask=mask), Keras may pass a list/tuple shape
        # for multi-input; here we always receive the sequence tensor shape.
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2 and not isinstance(input_shape[-1], int):
            seq_shape = input_shape[0]
        else:
            seq_shape = input_shape
        hidden_dim = int(seq_shape[-1])
        units = self.units or hidden_dim
        self.W = self.add_weight(
            name="attn_W",
            shape=(hidden_dim, units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.v = self.add_weight(
            name="attn_v",
            shape=(units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        # inputs: (batch, timesteps, hidden)
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1))
        score = tf.tensordot(score, self.v, axes=1)  # (batch, timesteps, 1)
        score = tf.squeeze(score, axis=-1)  # (batch, timesteps)

        if mask is not None:
            mask = tf.cast(mask, score.dtype)
            score = score + (1.0 - mask) * -1e9

        weights = tf.nn.softmax(score, axis=-1)  # (batch, timesteps)
        weights_exp = tf.expand_dims(weights, axis=-1)
        context = tf.reduce_sum(inputs * weights_exp, axis=1)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg
