import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class MixtureOfExperts(layers.Layer):
    def __init__(self, n_experts, n_classes, hidden=128, balance=0.05, **kw):
        super().__init__(**kw)
        self.n_experts, self.balance = n_experts, balance
        self.gate = layers.Dense(n_experts, activation="softmax", name="gate")
        self.experts = [tf.keras.Sequential([layers.Dense(hidden, activation="relu"),
                                             layers.Dense(n_classes)], name=f"expert_{i}")
                        for i in range(n_experts)]

    def call(self, x):
        g = self.gate(x)                                          # (batch, experts)
        e = tf.stack([ex(x) for ex in self.experts], axis=1)      # (batch, experts, classes)
        usage = tf.reduce_mean(g, axis=0)                         # average weight per expert
        self.add_loss(self.balance * self.n_experts * tf.reduce_sum(usage ** 2))
        return tf.einsum("be,bec->bc", g, e)                      # weighted sum of expert scores

def res_block(x, filters, downsample=True):
    stride = 2 if downsample else 1
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))

def build_moe(size, n_classes, n_experts=4, balance=0.05, augmentation=None):
    inp = tf.keras.Input((size, size, 3))
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)                           # 256 features per image
    out = MixtureOfExperts(n_experts, n_classes, balance=balance, name="moe")(x)
    m = tf.keras.Model(inp, out)
    m.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m

class GateMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, batches=20):
        super().__init__()
        self.val_ds, self.batches, self.usage, self.entropy = val_ds.take(batches), batches, [], []

    def on_train_begin(self, logs=None):
        moe = self.model.get_layer("moe")
        self.feat = tf.keras.Model(self.model.input, moe.input)

    def on_epoch_end(self, epoch, logs=None):
        moe = self.model.get_layer("moe")
        g = moe.gate(self.feat.predict(self.val_ds, verbose=0)).numpy()
        u = g.mean(0)
        self.usage.append(u)
        # 1.0 = perfectly even, 0.0 = one expert takes everything
        self.entropy.append(float(-(u * np.log(u + 1e-9)).sum() / np.log(len(u))))
        print(f"  usage {u.round(2)}  wins {np.bincount(g.argmax(1), minlength=len(u))}")

