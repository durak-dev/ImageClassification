"""The models, used like the prebuilt models of tf.keras.applications.

    from utils.moe_project import PMoEClassifier, compile_default

    model = PMoEClassifier(input_shape=(64, 64, 3), classes=200, n_experts=4, l=8)
    compile_default(model)                  # or your own model.compile(...)
    model.fit(train_ds, validation_data=val_ds, epochs=10)

Every class here IS a tf.keras.Model (a functional model), so summary(), fit(), evaluate(), predict(),
save_weights() and so on work as usual. Like the prebuilt Keras models they come back NOT compiled.
All of them output LOGITS (raw class scores), so compile them with from_logits=True.

    CNNClassifier    backbone + plain dense head (baseline)
    MoEClassifier    backbone + plain mixture of experts (MoE.py)
    PMoEClassifier   backbone + patch-level routing (pmoe.py)
    MGEClassifier    experts at three depths of the backbone + diversity penalty (mge.py)

Defaults come from config.py, but every value can be given as an argument instead.
"""
import tensorflow as tf
from tensorflow.keras import layers

from . import config
from .backbone import backbone, backbone_feature_map
from .moe import MixtureOfExperts
from .pmoe import PatchMoE
from .mge import mge_graph, mge_outputs

DEFAULT_INPUT_SHAPE = (config.IMG_SIZE, config.IMG_SIZE, 3)


def compile_default(model, lr=config.LR, weight_decay=config.WEIGHT_DECAY):
    """The compile settings used for every model in this project: AdamW, sparse cross-entropy on
    logits, accuracy. (Same as _compile in MoE.py.) Works for the multi-output MGE model too.
    Returns the model so you can write  model = compile_default(PMoEClassifier(...))."""
    model.compile(tf.keras.optimizers.AdamW(lr, weight_decay=weight_decay),
                  tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    return model


def _plain(value):
    """Tuples/lists/trackers -> plain lists, so get_config() can be written to JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return [_plain(v) for v in value]


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="moe_project")
class CNNClassifier(tf.keras.Model):
    """Baseline: the backbone and a plain dense head sized to match the MoE's hidden units
    (n_experts * hidden), so any difference comes from the mixture, not from capacity.
    Head: Dense(n_experts * hidden, relu) -> Dropout(head_drop) -> Dense(classes)."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES,
                 n_experts=config.N_EXPERTS, hidden=config.HIDDEN, augmentation=None,
                 drop=config.DROP, head_drop=config.EXPERT_DROP, name="cnn", **kwargs):
        inp = tf.keras.Input(input_shape)
        x = backbone(inp, augmentation, drop)
        x = layers.Dense(n_experts * hidden, activation="relu")(x)
        x = layers.Dropout(head_drop)(x)
        out = layers.Dense(classes)(x)
        super().__init__(inp, out, name=name, **kwargs)
        self._settings = dict(input_shape=input_shape, classes=classes, n_experts=n_experts,
                              hidden=hidden, drop=drop, head_drop=head_drop, name=name)

    def get_config(self):                       # the augmentation layer is not saved: pass it again
        return {k: _plain(v) for k, v in self._settings.items()}


# ---------------------------------------------------------------------------
# Plain mixture of experts
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="moe_project")
class MoEClassifier(tf.keras.Model):
    """Backbone -> pooled vector -> MixtureOfExperts (moe.py): a gate and n_experts small networks.
    The layer is named "moe"."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES,
                 n_experts=config.N_EXPERTS, hidden=config.HIDDEN, balance=config.BALANCE,
                 augmentation=None, drop=config.DROP, expert_drop=config.EXPERT_DROP,
                 name="moe_model", **kwargs):
        inp = tf.keras.Input(input_shape)
        x = backbone(inp, augmentation, drop)
        out = MixtureOfExperts(n_experts, classes, hidden=hidden, balance=balance, drop=expert_drop,
                               name="moe")(x)
        super().__init__(inp, out, name=name, **kwargs)
        self._settings = dict(input_shape=input_shape, classes=classes, n_experts=n_experts,
                              hidden=hidden, balance=balance, drop=drop, expert_drop=expert_drop, name=name)

    def get_config(self):
        return {k: _plain(v) for k, v in self._settings.items()}


# ---------------------------------------------------------------------------
# Patch-level routing
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable(package="moe_project")
class PMoEClassifier(tf.keras.Model):
    """Backbone -> feature map BEFORE pooling -> PatchMoE (pmoe.py). Each expert has its own router and
    keeps its best l patches. l must not exceed the number of patches (backbone.n_patches).
    The layer is named "pmoe" (expert_choices() looks for that name)."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES,
                 n_experts=config.N_EXPERTS, l=config.L_PATCHES, hidden=config.PMOE_HIDDEN,
                 augmentation=None, drop=config.DROP, expert_drop=config.EXPERT_DROP,
                 name="pmoe_model", **kwargs):
        inp = tf.keras.Input(input_shape)
        x = backbone_feature_map(inp, augmentation, drop)
        out = PatchMoE(n_experts, classes, l=l, hidden=hidden, drop=expert_drop, name="pmoe")(x)
        super().__init__(inp, out, name=name, **kwargs)
        self._settings = dict(input_shape=input_shape, classes=classes, n_experts=n_experts, l=l,
                              hidden=hidden, drop=drop, expert_drop=expert_drop, name=name)

    def get_config(self):
        return {k: _plain(v) for k, v in self._settings.items()}


# ---------------------------------------------------------------------------
# Mixture of granularity-specific experts
# ---------------------------------------------------------------------------

class _MultiOutputModel(tf.keras.Model):
    """A Keras model with several outputs that all get the SAME label (one output per expert + the ensemble).

    Plain Keras wants one loss, one metric list and one label per output. This class lets you use it like a
    one-output model instead:
      * compile(loss=..., metrics=[...])  : a single loss / metric list is repeated for every output
      * fit / evaluate with (image, label) data : the label is used for every output
    """

    def compile(self, optimizer="adam", loss=None, metrics=None, **kwargs):
        n = len(self.outputs)
        if loss is not None and not isinstance(loss, (list, tuple, dict)):
            loss = [loss] * n
        if metrics is not None and not isinstance(metrics, dict) and not any(isinstance(m, (list, tuple)) for m in metrics):
            metrics = [list(metrics) for _ in range(n)]
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def _labels_for_all_outputs(self, y):
        if y is not None and not isinstance(y, (list, tuple, dict)):
            return [y] * len(self.outputs)
        return y

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        return super().compute_loss(x, self._labels_for_all_outputs(y), y_pred, sample_weight, training)

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        return super().compute_metrics(x, self._labels_for_all_outputs(y), y_pred, sample_weight)


@tf.keras.utils.register_keras_serializable(package="moe_project")
class MGEClassifier(_MultiOutputModel):
    """MGE version 1: three experts at three depths of the backbone (64, 128 and 256 channels), trained
    together with the diversity penalty (mge.py).
    Outputs, in this order: ensemble, then expert_<i> for every i in `order`. The ensemble (average of the
    experts) is only for measuring; every expert is trained by its own cross-entropy.
    `order` says which expert counts as "earlier" in the penalty (0 = 64 ch, 1 = 128 ch, 2 = 256 ch).

    stages() gives MGE version 2 (the gradual schedule): a list of models that share their layers."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES,
                 diversity_weight=config.DIVERSITY_WEIGHT, direction=config.DIVERSITY_DIRECTION,
                 order=config.MGE_ORDER, augmentation=None, drop=config.DROP,
                 expert_drop=config.EXPERT_DROP, name="mge", **kwargs):
        inp, expert_scores = mge_graph(input_shape, classes, augmentation, drop, expert_drop)
        outputs = mge_outputs(expert_scores, list(order), diversity_weight, direction)
        super().__init__(inp, outputs, name=name, **kwargs)
        self._settings = dict(input_shape=input_shape, classes=classes, diversity_weight=diversity_weight,
                              direction=direction, order=order, drop=drop, expert_drop=expert_drop, name=name)

    def stages(self):
        """MGE version 2: a LIST of models, stage 1 with only the first expert of `order`, stage 2 with the
        first two, and so on (the last one has all experts, like this model). They share their layers
        with this model and with each other, so you train stage 1, then stage 2 on top of it, ...
        (train.run("mge_gradual", ...) does this for you). Compile every stage before training it.
        The one-at-a-time idea is the paper's; the order and "keep training the old experts" are my guesses."""
        order = list(self._settings["order"])
        expert_scores = [self.get_layer(f"expert_{i}_classifier").output for i in range(3)]
        stage_models = []
        for n_active in range(1, len(order) + 1):
            outputs = mge_outputs(expert_scores, order[:n_active],
                                  self._settings["diversity_weight"], self._settings["direction"])
            stage_models.append(_MultiOutputModel(self.input, outputs, name=f"{self.name}_stage{n_active}"))
        return stage_models

    def get_config(self):
        return {k: _plain(v) for k, v in self._settings.items()}
