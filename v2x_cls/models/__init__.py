"""Model builders for BRNN, CNN-BRNN, and CNN-BRNN + Attention."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

from v2x_cls.models.attention import AdditiveAttention, PaddingMask


def _dense_head(x, num_classes: int, dense_units: int, use_paper_dense_stack: bool):
    """Paper output stack: Dense(sigmoid) then Dense(softmax)."""
    if use_paper_dense_stack:
        x = layers.Dense(dense_units, activation="sigmoid", name="dense_sigmoid")(x)
    else:
        x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_brnn(
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embedding_dim: int = 64,
    rnn_units: int = 64,
    dense_units: int = 64,
    use_paper_dense_stack: bool = True,
    name: str = "brnn",
) -> Model:
    """Paper baseline: Embedding -> BiSimpleRNN(seq) -> Flatten -> Dense stack."""
    inputs = layers.Input(shape=(max_len,), name="input_ids")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(inputs)
    x = layers.Bidirectional(
        layers.SimpleRNN(rnn_units, return_sequences=True),
        name="bidirectional_rnn",
    )(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = _dense_head(x, num_classes, dense_units, use_paper_dense_stack)
    return Model(inputs, outputs, name=name)


def build_cnn_brnn(
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embedding_dim: int = 64,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
    rnn_units: int = 64,
    dense_units: int = 64,
    use_paper_dense_stack: bool = True,
    name: str = "cnn_brnn",
) -> Model:
    """Paper best model: Embedding -> Conv1D -> BiSimpleRNN(seq) -> Flatten -> Dense.

    Following Hassan & Mahmood (2017) as cited by the paper, a CNN front-end
    replaces pooling and feeds a bidirectional recurrent encoder.
    """
    inputs = layers.Input(shape=(max_len,), name="input_ids")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(inputs)
    x = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d",
    )(x)
    x = layers.Bidirectional(
        layers.SimpleRNN(rnn_units, return_sequences=True),
        name="bidirectional_rnn",
    )(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = _dense_head(x, num_classes, dense_units, use_paper_dense_stack)
    return Model(inputs, outputs, name=name)


def build_cnn_brnn_attention(
    vocab_size: int,
    num_classes: int,
    max_len: int,
    embedding_dim: int = 64,
    cnn_filters: int = 64,
    cnn_kernel_size: int = 3,
    rnn_units: int = 64,
    dense_units: int = 64,
    use_paper_dense_stack: bool = True,
    name: str = "cnn_brnn_attention",
) -> Tuple[Model, Model]:
    """Proposed model: Embedding -> Conv1D -> BiRNN(seq) -> Attention -> Dense.

    Returns:
      classifier: end-to-end classification model
      attention_model: same backbone outputting (probabilities, attention weights)
    """
    inputs = layers.Input(shape=(max_len,), name="input_ids")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embedding")(inputs)
    x = layers.Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d",
    )(x)
    seq = layers.Bidirectional(
        layers.SimpleRNN(rnn_units, return_sequences=True),
        name="bidirectional_rnn",
    )(x)
    # Conv1D does not reliably propagate Embedding masks, so build an explicit
    # padding mask from input token ids (0 = pad) for Attention.
    pad_mask = PaddingMask(name="padding_mask")(inputs)
    context, attn_weights = AdditiveAttention(name="attention")(seq, mask=pad_mask)
    outputs = _dense_head(context, num_classes, dense_units, use_paper_dense_stack)

    classifier = Model(inputs, outputs, name=name)
    attention_model = Model(inputs, [outputs, attn_weights], name=f"{name}_with_weights")
    return classifier, attention_model


def build_model_by_name(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    cfg: Dict[str, Any],
) -> Tuple[Model, Optional[Model]]:
    mcfg = cfg["model"]
    max_len = cfg["preprocessing"]["max_sequence_length"]
    common = dict(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=max_len,
        embedding_dim=mcfg["embedding_dim"],
        rnn_units=mcfg["rnn_units"],
        dense_units=mcfg["dense_units"],
        use_paper_dense_stack=mcfg["use_paper_dense_stack"],
    )

    if model_name == "brnn":
        return build_brnn(**common, name="brnn"), None
    if model_name == "cnn_brnn":
        return (
            build_cnn_brnn(
                **common,
                cnn_filters=mcfg["cnn_filters"],
                cnn_kernel_size=mcfg["cnn_kernel_size"],
                name="cnn_brnn",
            ),
            None,
        )
    if model_name in {"cnn_brnn_attention", "cnn_brnn_attn"}:
        clf, attn = build_cnn_brnn_attention(
            **common,
            cnn_filters=mcfg["cnn_filters"],
            cnn_kernel_size=mcfg["cnn_kernel_size"],
            name="cnn_brnn_attention",
        )
        return clf, attn
    raise ValueError(f"Unknown model_name: {model_name}")


def compile_model(model: Model, cfg: Dict[str, Any]) -> Model:
    tcfg = cfg["training"]
    optimizer = tf.keras.optimizers.Adam(learning_rate=tcfg["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss=tcfg["loss"],
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
