import tensorflow as tf
from tensorflow.keras import layers

class MixtureOfExperts(layers.Layer):
    def __init__(self, n_experts, n_classes, hidden=128, balance=0.01, **kw):
        super().__init__(**kw)
        self.n_experts, self.balance = n_experts, balance
        self.gate = layers.Dense(n_experts, activation="softmax", name="gate")
        self.experts = [tf.keras.Sequential([layers.Dense(hidden, activation="relu"),
                                             layers.Dense(n_classes)], name=f"expert_{i}")
                        for i in range(n_experts)]

    def call(self, x):
        g = self.gate(x)                                        # (B, E) routing weights
        e = tf.stack([ex(x) for ex in self.experts], axis=1)    # (B, E, C) logits per expert
        usage = tf.reduce_mean(g, axis=0)                       # average share per expert
        self.add_loss(self.balance * self.n_experts * tf.reduce_sum(usage ** 2))
        return tf.einsum("be,bec->bc", g, e)                    # gate weighted logits

def res_block(x, filters, downsample=True):
    stride = 2 if downsample else 1
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:       # match shapes so the two paths can be added
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))

def build_moe(size, n_classes, n_experts=8, augmentation=None, head_skip=False):
    inp = tf.keras.Input((size, size, 3))
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)    # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    x = res_block(x, 64)
    x = res_block(x, 128)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    out = MixtureOfExperts(n_experts, n_classes, name="moe")(x)
    if head_skip:                                                   # optional direct path beside the experts
        out = layers.Add()([out, layers.Dense(n_classes, name="shared_head")(x)])
    m = tf.keras.Model(inp, out)
    m.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m