import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import f1_score

# Every build_* function takes `backbone_fn(inp, augmentation) -> [s1, s2, s3]` (three feature maps, shallow -> deep,
# each (batch, H, W, C)); the notebooks pass their frozen saved backbone. CNN / MoE pool s3, pMoE reads s3 as patches,
# MGE uses all three.


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


def sparse_ce(n_classes, label_smoothing=0.0):
    """Cross-entropy on integer labels and logits, with optional label smoothing
    (Keras 3's SparseCategoricalCrossentropy has no label_smoothing)."""
    def loss(y_true, y_pred):
        y = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), n_classes)
        return tf.keras.losses.categorical_crossentropy(y, tf.cast(y_pred, tf.float32), from_logits=True,
                                                        label_smoothing=label_smoothing)
    return loss


def compile_head(m, lr, weight_decay):
    """The compile every head uses. lr: a float or a schedule. AdamW applies lr * weight_decay per step;
    biases and batch-norm parameters get no weight decay."""
    opt = tf.keras.optimizers.AdamW(lr, weight_decay=weight_decay)
    opt.exclude_from_weight_decay(var_names=["bias", "gamma", "beta"])
    m.compile(opt, tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return m


def build_moe(size, n_classes, backbone_fn, n_experts=4, balance=0.05, augmentation=None,
              drop=0.3, expert_drop=0.3, lr=1e-3, weight_decay=1e-4):
    """drop: dropout on the pooled s3 features."""
    inp = tf.keras.Input((size, size, 3))
    x = layers.GlobalAveragePooling2D()(backbone_fn(inp, augmentation)[-1])
    x = layers.Dropout(drop)(x)
    out = MixtureOfExperts(n_experts, n_classes, balance=balance, drop=expert_drop, name="moe")(x)
    return compile_head(tf.keras.Model(inp, out), lr, weight_decay)

def build_cnn(size, n_classes, backbone_fn, n_experts=4, hidden=128, augmentation=None,
              drop=0.3, head_drop=0.3, lr=1e-3, weight_decay=1e-4):
    """Baseline: same backbone, plain dense head sized to match the MoE's hidden units
    (n_experts * hidden), so any difference comes from the mixture, not from capacity. drop: dropout on the pooled s3."""
    inp = tf.keras.Input((size, size, 3))
    x = layers.GlobalAveragePooling2D()(backbone_fn(inp, augmentation)[-1])
    x = layers.Dropout(drop)(x)
    x = layers.Dense(n_experts * hidden, activation="relu")(x)
    x = layers.Dropout(head_drop)(x)
    out = layers.Dense(n_classes)(x)
    return compile_head(tf.keras.Model(inp, out), lr, weight_decay)

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
            pooled = tf.reduce_sum(h * gates[:, i, :, None], axis=1)   # gate-weighted average (gates already sum to 1)
            scores.append(self.out_layers[i](pooled))
        return tf.add_n(scores)                                       # experts are ADDED, as in the paper

def expert_choices(model, images, layer_name="pmoe"):
    """Which patches each expert picked. Returns (top_idx (batch, experts, l), gates, grid (H, W));
    patch p sits at row p // W, column p % W."""
    layer = model.get_layer(layer_name)
    feature_map = tf.keras.Model(model.input, layer.input).predict(images, verbose=0)
    _, top_idx, gates = layer.route(feature_map)
    return top_idx.numpy(), gates.numpy(), layer.grid

def build_pmoe(size, n_classes, backbone_fn, n_experts=4, l=8, hidden=64, augmentation=None, expert_drop=0.3,
               lr=1e-3, weight_decay=1e-4):
    inp = tf.keras.Input((size, size, 3))
    x = backbone_fn(inp, augmentation)[-1]                         # s3 as a feature map, NOT pooled
    out = PatchMoE(n_experts, n_classes, l, hidden=hidden, drop=expert_drop, name="pmoe")(x)
    return compile_head(tf.keras.Model(inp, out), lr, weight_decay)


# ---------------------------------------------------------------- MGE: one expert per backbone depth
# Zhang et al., ICCV 2019, adapted. UNVERIFIED guess: the diversity formula/direction/weight were not taken from the
# paper. Penalty = exp(-KL(p_earlier || p_later)) averaged over expert pairs.
class DiversityPenalty(layers.Layer):
    """Pass-through: adds weight * penalty to the loss and returns the expert scores unchanged."""
    def __init__(self, weight=0.1, direction="earlier_vs_later", **kw):
        super().__init__(**kw)
        self.weight, self.direction = weight, direction

    def call(self, expert_scores):
        log_p = [tf.nn.log_softmax(s, axis=-1) for s in expert_scores]
        values = []
        for a in range(len(log_p)):
            for b in range(a + 1, len(log_p)):
                ref, other = (log_p[a], log_p[b]) if self.direction == "earlier_vs_later" else (log_p[b], log_p[a])
                kl = tf.reduce_sum(tf.exp(ref) * (ref - other), axis=-1)       # KL(ref || other), (batch,)
                values.append(tf.reduce_mean(tf.exp(-kl)))
        self.add_loss(self.weight * tf.add_n(values) / len(values))
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

def build_mge(size, n_classes, backbone_fn, diversity_weight=0.1, direction="earlier_vs_later", augmentation=None,
              expert_drop=0.3, lr=1e-3, weight_decay=1e-4, proj=256):
    """Outputs: ensemble, expert_0 (shallowest), expert_1, expert_2 (deepest). Each expert pools its own map.
    proj: each expert first applies a trainable 1x1 conv (proj channels) + BN + ReLU to its map, so the shallow
    experts are not limited to the raw average of a 32/64-channel map. proj=None: no projection (old behaviour)."""
    inp = tf.keras.Input((size, size, 3))
    scores = []
    for fmap in backbone_fn(inp, augmentation):
        if proj:
            fmap = layers.Conv2D(proj, 1, use_bias=False)(fmap)
            fmap = layers.ReLU()(layers.BatchNormalization()(fmap))
        x = layers.GlobalAveragePooling2D()(fmap)
        x = layers.Dropout(expert_drop)(x)
        scores.append(layers.Dense(n_classes)(x))
    scores = DiversityPenalty(diversity_weight, direction, name="diversity")(scores)
    outs = [EnsembleAverage(name="ensemble")(scores)] + [layers.Identity(name=f"expert_{i}")(s) for i, s in enumerate(scores)]
    return compile_head(MultiOutputModel(inp, outs, name="mge"), lr, weight_decay)


class GateMonitor(tf.keras.callbacks.Callback):
    """MoE only. After every epoch: the average gate weight per expert on val_ds, written into the epoch logs as
    gate_usage_<i>, plus gate_balance (1.0 = perfectly even, 0.0 = one expert takes everything). Put it before
    CSVLogger, so these columns land in the CSV log."""
    def __init__(self, val_ds, batches=20):
        super().__init__()
        self.val_ds = val_ds.take(batches)

    def on_train_begin(self, logs=None):
        self.feat = tf.keras.Model(self.model.input, self.model.get_layer("moe").input)

    def on_epoch_end(self, epoch, logs=None):
        g = self.model.get_layer("moe").gate(self.feat.predict(self.val_ds, verbose=0)).numpy()
        u = g.mean(0)
        balance = float(-(u * np.log(u + 1e-9)).sum() / np.log(len(u)))
        if logs is not None:
            logs.update({f"gate_usage_{i}": float(v) for i, v in enumerate(u)}, gate_balance=balance)
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
