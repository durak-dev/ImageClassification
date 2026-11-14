"""The four model builders (+ the optional gradual MGE builder).

All of them share the same style:
    build_xxx(size, n_classes, <model settings>, augmentation=None, drop=0.1, lr=..., weight_decay=...)
and all of them end in the SAME _compile() (AdamW, sparse cross-entropy on logits, accuracy).
Defaults come from config.py, but every value can be given as an argument instead.
"""
import tensorflow as tf
from tensorflow.keras import layers

from . import config
from .backbone import backbone, backbone_feature_map, n_patches
from .moe import MixtureOfExperts
from .pmoe import PatchMoE
from .mge import mge_graph, mge_model


def _compile(m, lr, weight_decay, n_outputs=1):
    """Copied from MoE.py. n_outputs=1 behaves exactly as there.
    n_outputs > 1 is only for the MGE model (one output per expert + the ensemble): Keras then needs
    the loss and the metric given once PER OUTPUT, so the same loss/metric is simply repeated."""
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if n_outputs == 1:
        m.compile(tf.keras.optimizers.AdamW(lr, weight_decay=weight_decay), loss, metrics=["accuracy"])
    else:
        m.compile(tf.keras.optimizers.AdamW(lr, weight_decay=weight_decay),
                  [loss] * n_outputs, metrics=[["accuracy"] for _ in range(n_outputs)])
    return m


def build_cnn(size, n_classes, n_experts=config.N_EXPERTS, hidden=config.HIDDEN, augmentation=None,
              drop=config.DROP, head_drop=config.EXPERT_DROP, lr=config.LR, weight_decay=config.WEIGHT_DECAY):
    """Baseline: same backbone, plain dense head sized to match the MoE's hidden units
    (n_experts * hidden), so any difference comes from the mixture, not from capacity."""
    inp = tf.keras.Input((size, size, 3))
    x = backbone(inp, augmentation, drop)
    x = layers.Dense(n_experts * hidden, activation="relu")(x)
    x = layers.Dropout(head_drop)(x)
    out = layers.Dense(n_classes)(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)


def build_moe(size, n_classes, n_experts=config.N_EXPERTS, hidden=config.HIDDEN, balance=config.BALANCE,
              augmentation=None, drop=config.DROP, expert_drop=config.EXPERT_DROP,
              lr=config.LR, weight_decay=config.WEIGHT_DECAY):
    """Plain mixture of experts on the pooled features (MoE.py). `hidden` is new here: in MoE.py it was
    fixed at the layer's default (128); the default value here is the same."""
    inp = tf.keras.Input((size, size, 3))
    x = backbone(inp, augmentation, drop)
    out = MixtureOfExperts(n_experts, n_classes, hidden=hidden, balance=balance, drop=expert_drop,
                           name="moe")(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)


def build_pmoe(size, n_classes, n_experts=config.N_EXPERTS, l=config.L_PATCHES, hidden=config.PMOE_HIDDEN,
               augmentation=None, drop=config.DROP, expert_drop=config.EXPERT_DROP,
               lr=config.LR, weight_decay=config.WEIGHT_DECAY):
    """Patch-level routing (pmoe.py) on the feature map before pooling.
    l = patches kept per expert; it must not exceed n_patches(size)."""
    if l > n_patches(size):
        raise ValueError(f"l={l} is more than the {n_patches(size)} patches an image of size {size} gives.")
    inp = tf.keras.Input((size, size, 3))
    x = backbone_feature_map(inp, augmentation, drop)
    out = PatchMoE(n_experts, n_classes, l=l, hidden=hidden, drop=expert_drop, name="pmoe")(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)


def build_mge(size, n_classes, diversity_weight=config.DIVERSITY_WEIGHT,
              direction=config.DIVERSITY_DIRECTION, order=config.MGE_ORDER, augmentation=None,
              drop=config.DROP, expert_drop=config.EXPERT_DROP, lr=config.LR, weight_decay=config.WEIGHT_DECAY):
    """MGE version 1: all three depth experts trained together, with the diversity penalty.
    Outputs (in this order): ensemble, then expert_<i> for every i in `order`.
    `order` also decides which expert counts as "earlier" in the penalty."""
    inp, expert_scores = mge_graph(size, n_classes, augmentation, drop, expert_drop)
    model = mge_model(inp, expert_scores, active=list(order),
                      diversity_weight=diversity_weight, direction=direction)
    return _compile(model, lr, weight_decay, n_outputs=len(model.outputs))


def build_mge_gradual(size, n_classes, diversity_weight=config.DIVERSITY_WEIGHT,
                      direction=config.DIVERSITY_DIRECTION, order=config.MGE_ORDER, augmentation=None,
                      drop=config.DROP, expert_drop=config.EXPERT_DROP, lr=config.LR,
                      weight_decay=config.WEIGHT_DECAY):
    """MGE version 2: returns a LIST of compiled models, one per stage. Stage 1 has only the first
    expert of `order`, stage 2 the first two, and so on. They SHARE their layers (and so their weights),
    so you train stage 1, then stage 2 on top of it, ... (train.run does this for you).
    The one-at-a-time idea is the paper's; the order and "keep training the old experts" are my guesses."""
    inp, expert_scores = mge_graph(size, n_classes, augmentation, drop, expert_drop)
    stages = []
    for n_active in range(1, len(order) + 1):
        model = mge_model(inp, expert_scores, active=list(order[:n_active]),
                          diversity_weight=diversity_weight, direction=direction)
        stages.append(_compile(model, lr, weight_decay, n_outputs=len(model.outputs)))
    return stages
