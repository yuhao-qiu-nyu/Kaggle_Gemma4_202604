"""
Model architecture + ensemble loader — ported from the 1st-place notebook.

Uses tf_keras (Keras 2) for full compatibility with the original .h5 weights.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocess import (
    CHANNELS,
    MAX_LEN,
    NUM_CLASSES,
    prepare_single_sample,
)

# ---------------------------------------------------------------------------
# Custom layers (Keras 2 / tf_keras)
# ---------------------------------------------------------------------------

class ECA(keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class LateDropout(keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.start_step = start_step
        self.dropout = keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(
            self._train_counter < self.start_step,
            lambda: inputs,
            lambda: self.dropout(inputs, training=training),
        )
        if training:
            self._train_counter.assign_add(1)
        return x


class CausalDWConv1D(keras.layers.Layer):
    def __init__(self, kernel_size=17, dilation_rate=1, use_bias=False,
                 depthwise_initializer="glorot_uniform", name="", **kwargs):
        super().__init__(name=name, **kwargs)
        self.causal_pad = keras.layers.ZeroPadding1D((dilation_rate * (kernel_size - 1), 0), name=name + "_pad")
        self.dw_conv = keras.layers.DepthwiseConv1D(
            kernel_size, strides=1, dilation_rate=dilation_rate,
            padding="valid", use_bias=use_bias,
            depthwise_initializer=depthwise_initializer, name=name + "_dwconv",
        )
        self.supports_masking = True

    def call(self, inputs):
        return self.dw_conv(self.causal_pad(inputs))


class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = dim ** -0.5
        self.num_heads = num_heads
        self.qkv = keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = keras.layers.Dropout(dropout)
        self.proj = keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = keras.layers.Permute((2, 1, 3))(
            keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv)
        )
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        if mask is not None:
            mask = mask[:, None, None, :]
        attn = keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)
        x = attn @ v
        x = keras.layers.Reshape((-1, self.dim))(keras.layers.Permute((2, 1, 3))(x))
        return self.proj(x)


# ---------------------------------------------------------------------------
# Block builders
# ---------------------------------------------------------------------------

def _conv1d_block(channel_size, kernel_size, dilation_rate=1, drop_rate=0.0,
                  expand_ratio=2, activation="swish", name=None):
    if name is None:
        name = str(keras.backend.get_uid("mbblock"))

    def apply(inputs):
        channels_in = keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio
        skip = inputs
        x = keras.layers.Dense(channels_expand, use_bias=True, activation=activation, name=name + "_expand_conv")(inputs)
        x = CausalDWConv1D(kernel_size, dilation_rate=dilation_rate, use_bias=False, name=name + "_dwconv")(x)
        x = keras.layers.BatchNormalization(momentum=0.95, name=name + "_bn")(x)
        x = ECA()(x)
        x = keras.layers.Dense(channel_size, use_bias=True, name=name + "_project_conv")(x)
        if drop_rate > 0:
            x = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "_drop")(x)
        if channels_in == channel_size:
            x = keras.layers.add([x, skip], name=name + "_add")
        return x

    return apply


def _transformer_block(dim=256, num_heads=4, expand=4, attn_dropout=0.2,
                        drop_rate=0.2, activation="swish"):
    def apply(inputs):
        x = keras.layers.BatchNormalization(momentum=0.95)(inputs)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = keras.layers.Add()([inputs, x])
        attn_out = x
        x = keras.layers.BatchNormalization(momentum=0.95)(x)
        x = keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = keras.layers.Dense(dim, use_bias=False)(x)
        x = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = keras.layers.Add()([attn_out, x])
        return x

    return apply


# ---------------------------------------------------------------------------
# Build Keras model graph
# ---------------------------------------------------------------------------

def build_model(max_len=MAX_LEN, dropout_step=0, dim=192):
    inp = keras.Input((max_len, CHANNELS))
    x = inp
    ksize = 17
    x = keras.layers.Dense(dim, use_bias=False, name="stem_conv")(x)
    x = keras.layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)

    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _transformer_block(dim, expand=2)(x)

    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
    x = _transformer_block(dim, expand=2)(x)

    if dim == 384:
        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _transformer_block(dim, expand=2)(x)

        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _conv1d_block(dim, ksize, drop_rate=0.2)(x)
        x = _transformer_block(dim, expand=2)(x)

    x = keras.layers.Dense(dim * 2, activation=None, name="top_conv")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = LateDropout(0.8, start_step=dropout_step)(x)
    x = keras.layers.Dense(NUM_CLASSES, name="classifier")(x)
    return keras.Model(inp, x)


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class ASLEnsemble:
    """Loads N weight files, runs ensemble-averaged inference."""

    def __init__(self, weight_paths: List[str], sign_map_path: str):
        self.s2p_map = {k.lower(): v for k, v in self._load_json(sign_map_path).items()}
        self.p2s_map = {v: k for k, v in self.s2p_map.items()}

        self.models: List[keras.Model] = []
        for p in weight_paths:
            m = build_model()
            # 强行加载，忽略那些由于 Keras 版本差异导致的多余变量（如 dropout 的计数器）
            try:
                # 尝试最宽容的加载方式
                m.load_weights(p, by_name=True, skip_mismatch=True)
            except TypeError:
                # 如果你的环境不支持 skip_mismatch 参数，则使用这种手动逻辑
                import h5py
                with h5py.File(p, 'r') as f:
                    # 这种方式会只针对名字匹配且形状一致的层进行填充
                    for layer in m.layers:
                        layer_name = layer.name
                        if layer_name in f:
                            print(f"尝试加载层权重: {layer_name}")
                            try:
                                # 手动设置权重，忽略不匹配的层
                                weights = [f[layer_name][p] for p in f[layer_name].attrs['weight_names']]
                                layer.set_weights(weights)
                            except:
                                print(f"⚠️ 跳过不匹配的层: {layer_name}")
            
            self.models.append(m)
            print(f"✅ 模型 {os.path.basename(p)} 权重加载逻辑处理完成")

    @staticmethod
    def _load_json(path: str):
        with open(path) as f:
            return json.load(f)

    def predict(self, raw_landmarks: np.ndarray, topk: int = 5) -> dict:
        """
        raw_landmarks: (T, 543, 3) float32
        Returns a product-ready dict.
        """
        x = prepare_single_sample(raw_landmarks)

        logits_stack = [m(x, training=False) for m in self.models]
        logits = tf.reduce_mean(tf.stack(logits_stack, axis=0), axis=0)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        top_ids = np.argsort(probs)[::-1][:topk]
        topk_predictions = [
            {"rank": i + 1, "label": self.p2s_map[int(cid)], "prob": float(probs[cid])}
            for i, cid in enumerate(top_ids)
        ]

        best_prob = topk_predictions[0]["prob"]
        if best_prob >= 0.80:
            confidence = "high"
        elif best_prob >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "predicted_label": topk_predictions[0]["label"],
            "predicted_prob": best_prob,
            "confidence": confidence,
            "topk_predictions": topk_predictions,
        }
