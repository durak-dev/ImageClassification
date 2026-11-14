"""The plain Mixture of Experts layer, copied from MoE.py (behaviour unchanged).

One gate looks at the pooled feature vector and decides how much to trust each expert.
Every expert sees the WHOLE vector.  (pmoe.py is the other idea: experts see different patches.)
"""
import tensorflow as tf
from tensorflow.keras import layers


class MixtureOfExperts(layers.Layer):
    def __init__(self, n_experts, n_classes, hidden=128, balance=0.05, drop=0.0, **kw):
        super().__init__(**kw)
        self.n_experts, self.balance = n_experts, balance
        self.gate = layers.Dense(n_experts, activation="softmax", name="gate")
        self.experts = [tf.keras.Sequential([layers.Dense(hidden, activation="relu"), layers.Dropout(drop),
                                             layers.Dense(n_classes)], name=f"expert_{i}")
                        for i in range(n_experts)]

    def call(self, x):
        g = self.gate(x)                                          # (batch, experts)
        e = tf.stack([ex(x) for ex in self.experts], axis=1)      # (batch, experts, classes)
        usage = tf.reduce_mean(g, axis=0)                         # average weight per expert
        self.add_loss(self.balance * self.n_experts * tf.reduce_sum(usage ** 2))
        return tf.einsum("be,bec->bc", g, e)                      # weighted sum of expert scores
