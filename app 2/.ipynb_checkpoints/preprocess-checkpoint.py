"""
Landmark preprocessing — ported from the 1st-place notebook.

Input:  (T, 543, 3)  raw MediaPipe landmarks
Output: (1, MAX_LEN, CHANNELS)  model-ready tensor
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

ROWS_PER_FRAME = 543
MAX_LEN = 384
NUM_CLASSES = 250
PAD = -100.0

NOSE = [1, 2, 98, 327]
LIP = [
    0, 61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)       # 118
CHANNELS = 6 * NUM_NODES               # 708


def _tf_nan_mean(x, axis=0, keepdims=False):
    return (
        tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims)
        / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)
    )


def _tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = _tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(_tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

from tensorflow.keras import layers

class Preprocess(layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks or POINT_LANDMARKS

    def call(self, inputs):
        x = inputs[None, ...] if tf.rank(inputs) == 3 else inputs

        mean = _tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = _tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)
        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, : self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        dx = tf.cond(
            tf.shape(x)[1] > 1,
            lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )
        dx2 = tf.cond(
            tf.shape(x)[1] > 2,
            lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]),
            lambda: tf.zeros_like(x),
        )

        n_pts = len(self.point_landmarks)
        x = tf.concat(
            [
                tf.reshape(x, (-1, length, 2 * n_pts)),
                tf.reshape(dx, (-1, length, 2 * n_pts)),
                tf.reshape(dx2, (-1, length, 2 * n_pts)),
            ],
            axis=-1,
        )
        return tf.where(tf.math.is_nan(x), tf.constant(0.0, x.dtype), x)


_preprocess_layer = Preprocess(max_len=MAX_LEN)


def prepare_single_sample(raw_landmarks: np.ndarray) -> tf.Tensor:
    # 1. 基础转换
    x = tf.convert_to_tensor(raw_landmarks, dtype=tf.float32)
    
    # 2. 预处理层（处理空数据的关键点）
    x = _preprocess_layer(x)
    
    # --- 新增安全检查 ---
    # 如果处理后没有帧（例如没抓到手），构造一个全 0 帧防止报错
    if tf.shape(x)[0] == 0:
        x = tf.zeros((1, CHANNELS), dtype=tf.float32)
    else:
        x = x[0] # 原有的逻辑
    # ------------------

    x = tf.expand_dims(x, axis=0) # (1, T', CHANNELS)
    
    seq_len = tf.shape(x)[1]
    if seq_len < MAX_LEN:
        # 补齐逻辑
        pad_tensor = tf.ones((1, MAX_LEN - seq_len, CHANNELS), dtype=x.dtype) * PAD
        x = tf.concat([x, pad_tensor], axis=1)
    else:
        # 裁剪逻辑
        x = x[:, :MAX_LEN, :]
        
    return x
