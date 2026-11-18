import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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

def res_block(x, filters, downsample=True, l2=0.0):
    """Residual block. l2 > 0 adds an L2 weight penalty on every conv kernel of the block."""
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    stride = 2 if downsample else 1
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False, kernel_regularizer=reg)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False, kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))

def backbone(inp, augmentation=None, drop=0.1):
    """Shared conv feature extractor -> 256 features. Used by BOTH models so the comparison is fair."""
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    x = res_block(x, 64)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 128)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dropout(drop * 2.5)(x)                             # feature dropout

def backbone_maps(inp, augmentation=None, drop=0.1):
    """Same layers as backbone(), but returns the 3 stage outputs [64, 128, 256 channels] (batch, H, W, C)."""
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    s1 = res_block(x, 64)
    s2 = res_block(layers.SpatialDropout2D(drop)(s1), 128)
    s3 = res_block(layers.SpatialDropout2D(drop)(s2), 256)
    return [s1, s2, s3]

def get_maps(inp, backbone_fn=None, augmentation=None, drop=0.1):
    """MGE: the 3 feature maps (shallow -> deep). backbone_fn=None: the default backbone above.
    Your backbone_fn(inp, augmentation) must return a list of 3 maps (batch, H, W, C); it applies the
    augmentation, rescaling and dropout itself, like backbone() does."""
    if backbone_fn is None:
        return backbone_maps(inp, augmentation, drop)
    maps = backbone_fn(inp, augmentation)
    assert isinstance(maps, (list, tuple)) and len(maps) == 3 and all(len(m.shape) == 4 for m in maps),         "the MGE backbone must return a list of 3 feature maps, each (batch, H, W, C), shallow -> deep"
    return maps

def get_map(inp, backbone_fn=None, augmentation=None, drop=0.1):
    """pMoE: ONE feature map (batch, H, W, C), not pooled. backbone_fn(inp, augmentation) returns it."""
    if backbone_fn is None:
        return backbone_maps(inp, augmentation, drop)[-1]
    x = backbone_fn(inp, augmentation)
    assert len(x.shape) == 4, f"the pMoE backbone must return a feature map (batch, H, W, C), got {tuple(x.shape)}"
    return x

def features(inp, backbone_fn=None, augmentation=None, drop=0.1):
    """CNN / MoE: a pooled feature vector (batch, F). backbone_fn(inp, augmentation) may return the vector
    itself (like backbone()) or a feature map, which then gets global-average-pooled."""
    if backbone_fn is None:
        return backbone(inp, augmentation, drop)
    x = backbone_fn(inp, augmentation)
    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D()(x)
    assert len(x.shape) == 2, f"the CNN/MoE backbone must return a vector (batch, F) or a map, got {tuple(x.shape)}"
    return x

def _compile(m, lr, weight_decay):
    m.compile(tf.keras.optimizers.AdamW(lr, weight_decay=weight_decay),
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
    return m

def build_moe(size, n_classes, n_experts=4, balance=0.05, augmentation=None,
              drop=0.1, expert_drop=0.3, lr=1e-3, weight_decay=1e-4, backbone_fn=None):
    inp = tf.keras.Input((size, size, 3))
    x = features(inp, backbone_fn, augmentation, drop)
    out = MixtureOfExperts(n_experts, n_classes, balance=balance, drop=expert_drop, name="moe")(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)

def build_cnn(size, n_classes, n_experts=4, hidden=128, augmentation=None,
              drop=0.1, head_drop=0.3, lr=1e-3, weight_decay=1e-4, backbone_fn=None):
    """Baseline: same backbone, plain dense head sized to match the MoE's hidden units
    (n_experts * hidden), so any difference comes from the mixture, not from capacity."""
    inp = tf.keras.Input((size, size, 3))
    x = features(inp, backbone_fn, augmentation, drop)
    x = layers.Dense(n_experts * hidden, activation="relu")(x)
    x = layers.Dropout(head_drop)(x)
    out = layers.Dense(n_classes)(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)

# ---------------------------------------------------------------- pMoE: patch-level routing
# (Chowdhury et al., ICML 2023). Every expert has its OWN router vector, scores all patches of the
# feature map (batch, H, W, C) and keeps its best l. Expert outputs are added. An adaptation, not a reproduction.
class PatchMoE(layers.Layer):
    def __init__(self, n_experts, n_classes, l, hidden=64, drop=0.0, **kw):
        super().__init__(**kw)
        self.n_experts, self.n_classes, self.l, self.hidden, self.drop = n_experts, n_classes, l, hidden, drop

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.grid, self.n_patches, self.channels = (h, w), h * w, c
        if self.l > self.n_patches:
            raise ValueError(f"l={self.l} but the feature map only has {self.n_patches} patches ({h}x{w})")
        self.router = self.add_weight(name="router", shape=(self.n_experts, c),
                                      initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.hidden_layers = [layers.Dense(self.hidden, activation="relu") for _ in range(self.n_experts)]
        self.drop_layers = [layers.Dropout(self.drop) for _ in range(self.n_experts)]
        self.out_layers = [layers.Dense(self.n_classes) for _ in range(self.n_experts)]
        for i in range(self.n_experts):
            self.hidden_layers[i].build((None, self.l, c))
            self.out_layers[i].build((None, self.hidden))
        super().build(input_shape)

    def route(self, feature_map):
        """patches (batch, n, C), top_idx (batch, experts, l) = patches kept by each expert, gates (batch, experts, l)."""
        patches = tf.reshape(feature_map, (tf.shape(feature_map)[0], self.n_patches, self.channels))
        scores = tf.einsum("bnc,kc->bkn", patches, self.router)       # (batch, experts, n)
        top_scores, top_idx = tf.math.top_k(scores, k=self.l)         # best l patches PER expert
        return patches, top_idx, tf.nn.softmax(top_scores, axis=-1)   # softmax over the kept scores only

    def call(self, feature_map, training=None):
        patches, top_idx, gates = self.route(feature_map)
        chosen = tf.gather(patches, top_idx, batch_dims=1)            # (batch, experts, l, C)
        scores = []
        for i in range(self.n_experts):
            h = self.drop_layers[i](self.hidden_layers[i](chosen[:, i]), training=training)   # (batch, l, hidden)
            pooled = tf.reduce_sum(h * gates[:, i, :, None], axis=1) / self.l                  # gate-weighted, 1/l
            scores.append(self.out_layers[i](pooled))
        return tf.add_n(scores)                                       # experts are ADDED, as in the paper

def expert_choices(model, images, layer_name="pmoe"):
    """Which patches each expert picked. Returns (top_idx (batch, experts, l), gates, grid (H, W));
    patch p sits at row p // W, column p % W."""
    layer = model.get_layer(layer_name)
    feature_map = tf.keras.Model(model.input, layer.input).predict(images, verbose=0)
    _, top_idx, gates = layer.route(feature_map)
    return top_idx.numpy(), gates.numpy(), layer.grid

def build_pmoe(size, n_classes, n_experts=4, l=8, hidden=64, augmentation=None, drop=0.1, expert_drop=0.3,
               lr=1e-3, weight_decay=1e-4, backbone_fn=None):
    inp = tf.keras.Input((size, size, 3))
    x = get_map(inp, backbone_fn, augmentation, drop)                # feature map, NOT pooled
    out = PatchMoE(n_experts, n_classes, l, hidden=hidden, drop=expert_drop, name="pmoe")(x)
    return _compile(tf.keras.Model(inp, out), lr, weight_decay)


# ---------------------------------------------------------------- MGE: one expert per backbone depth
# Zhang et al., ICCV 2019, adapted. UNVERIFIED guess: the diversity formula/direction/weight were not taken from the
# paper (see utils/moe_project/mge.py). Penalty = exp(-KL(p_earlier || p_later)) averaged over expert pairs.
def diversity_penalty_value(expert_scores, direction="earlier_vs_later"):
    log_p = [tf.nn.log_softmax(s, axis=-1) for s in expert_scores]
    values = []
    for a in range(len(log_p)):
        for b in range(a + 1, len(log_p)):
            ref, other = (log_p[a], log_p[b]) if direction == "earlier_vs_later" else (log_p[b], log_p[a])
            kl = tf.reduce_sum(tf.exp(ref) * (ref - other), axis=-1)       # KL(ref || other), (batch,)
            values.append(tf.reduce_mean(tf.exp(-kl)))
    return tf.add_n(values) / len(values)

class DiversityPenalty(layers.Layer):
    """Pass-through: adds weight * penalty to the loss and returns the expert scores unchanged."""
    def __init__(self, weight=0.1, direction="earlier_vs_later", **kw):
        super().__init__(**kw)
        self.weight, self.direction = weight, direction

    def call(self, expert_scores):
        self.add_loss(self.weight * diversity_penalty_value(expert_scores, self.direction))
        return [tf.identity(s) for s in expert_scores]

class EnsembleAverage(layers.Layer):
    """Average of the experts' scores, only to MEASURE the combination (stop_gradient: each expert has its own loss)."""
    def call(self, expert_scores):
        return tf.stop_gradient(tf.add_n(expert_scores) / len(expert_scores))

class MultiOutputModel(tf.keras.Model):
    """Several outputs, ONE label: compile(loss, metrics) and fit((image, label)) work as for a one-output model."""
    def compile(self, optimizer="adam", loss=None, metrics=None, **kw):
        n = len(self.outputs)
        super().compile(optimizer=optimizer, loss=[loss] * n, metrics=[list(metrics) for _ in range(n)], **kw)

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        return super().compute_loss(x, [y] * len(self.outputs), y_pred, sample_weight, training)

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        return super().compute_metrics(x, [y] * len(self.outputs), y_pred, sample_weight)

def build_mge(size, n_classes, diversity_weight=0.1, direction="earlier_vs_later", augmentation=None, drop=0.1,
              expert_drop=0.3, lr=1e-3, weight_decay=1e-4, backbone_fn=None):
    """Outputs: ensemble, expert_0 (shallowest), expert_1, expert_2 (deepest). Each expert pools its own map."""
    inp = tf.keras.Input((size, size, 3))
    scores = []
    for i, fmap in enumerate(get_maps(inp, backbone_fn, augmentation, drop)):
        x = layers.GlobalAveragePooling2D()(fmap)
        x = layers.Dropout(expert_drop)(x)
        scores.append(layers.Dense(n_classes)(x))
    scores = DiversityPenalty(diversity_weight, direction, name="diversity")(scores)
    outs = [EnsembleAverage(name="ensemble")(scores)] + [layers.Identity(name=f"expert_{i}")(s) for i, s in enumerate(scores)]
    return _compile(MultiOutputModel(inp, outs, name="mge"), lr, weight_decay)


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


class MacroF1(tf.keras.callbacks.Callback):
    """Computes macro F1 on val_ds at the end of each epoch; stores it in logs['val_f1']."""
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.y_true = np.concatenate([y.numpy() for _, y in val_ds])   # labels fixed, val not shuffled

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_ds, verbose=0)
        if isinstance(pred, (list, tuple)):          # MGE has several outputs: score the first (the ensemble)
            pred = pred[0]
        y_pred = pred.argmax(1)
        f1 = f1_score(self.y_true, y_pred, average="macro")
        if logs is not None:
            logs["val_f1"] = f1                  # lands in history.history
        print(f"  val macro-F1 {f1:.4f}")


def default_callbacks(val_ds, name="model", log_dir="history_logs", patience=8, gate=False):
    """Standard callbacks for fit(): val macro-F1, a per-epoch history CSV
    (log_dir/<name>.csv, survives a crashed Colab session) and early stopping.
    gate=True also adds GateMonitor (MoE only). Returns (callbacks, gate_monitor or None).
    MacroF1 must come before CSVLogger so that val_f1 ends up in the CSV."""
    os.makedirs(log_dir, exist_ok=True)
    mon = GateMonitor(val_ds) if gate else None
    cbs = ([mon] if gate else []) + [
        MacroF1(val_ds),
        tf.keras.callbacks.CSVLogger(os.path.join(log_dir, f"{name}.csv")),
        tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)]
    return cbs, mon


