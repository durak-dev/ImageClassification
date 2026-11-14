"""MGE-style experts: a mixture of GRANULARITY-specific experts.
Inspired by Zhang et al., ICCV 2019, "Learning a Mixture of Granularity Specific Experts
for Fine Grained Categorization".

!! READ THIS FIRST - what is verified and what is a guess !!
I could NOT read the paper text (the official PDF refused automated download), so:
  * VERIFIED against nothing: the exact diversity formula, its direction, the weight lambda,
    and the order / schedule of the gradual training. They were NOT taken from the paper.
  * What the diversity penalty below is based on: a community PyTorch re-implementation of the
    paper (github.com/kiminh/MGE-CNN, not checked to be the authors' code). It uses, between two experts' predicted
    class distributions,    exp( -KL(p_earlier || p_later) )    added to the loss, so minimising the
    loss makes the KL large (experts disagree). That code also (a) blanks out the true class before the
    KL, which needs the labels - impossible inside add_loss, so it is left out here - and (b) only
    compares each expert with the previous one, I compare all pairs.
  * The paper's "granularity" is about image scale (whole image, then a cropped object box, then a
    smaller crop in that implementation). Attaching experts at different DEPTHS of the backbone,
    as you asked, is a different (and simpler) notion of granularity: an adaptation.
  Please compare diversity_penalty_value() with the paper's equation before trusting the numbers.

What is here
  mge_graph()          backbone + 3 experts (one per depth). Layers are created ONCE.
  DiversityPenalty     the add_loss penalty (version 1 and 2 both use it)
  EnsembleAverage      average of the experts' scores, used only for evaluation
  mge_model()          wires a chosen subset of experts into a Model (version 1 = all, version 2 = growing subset)
"""
import tensorflow as tf
from tensorflow.keras import layers

from .backbone import backbone_multi_depth


def mge_expert(feature_map, n_classes, expert_drop, index):
    """One expert: its OWN global pooling and its OWN class output layer on one backbone depth."""
    x = layers.GlobalAveragePooling2D(name=f"expert_{index}_pool")(feature_map)
    x = layers.Dropout(expert_drop, name=f"expert_{index}_drop")(x)
    return layers.Dense(n_classes, name=f"expert_{index}_classifier")(x)     # scores (logits)


def mge_graph(size, n_classes, augmentation=None, drop=0.1, expert_drop=0.3):
    """Backbone + one expert at each of the three depths (64, 128, 256 channels).
    Returns (input, [scores_0, scores_1, scores_2]); expert 0 is the shallowest, 2 the deepest."""
    inp = tf.keras.Input((size, size, 3))
    stages = backbone_multi_depth(inp, augmentation, drop)
    expert_scores = [mge_expert(stage, n_classes, expert_drop, i) for i, stage in enumerate(stages)]
    return inp, expert_scores


def diversity_penalty_value(expert_scores, direction="earlier_vs_later"):
    """The penalty as a plain function (so you can test it on its own).
    expert_scores: list of tensors (batch, classes), IN THE ORDER the experts were added.
    For every pair (earlier a, later b):   exp( -KL(p_a || p_b) )   averaged over the batch,
    then averaged over the pairs.  KL(p || q) = sum_c p_c * (log p_c - log q_c).
    Small KL (experts agree) -> value near 1. Large KL (experts disagree) -> value near 0.
    direction = "earlier_vs_later" -> KL(p_earlier || p_later)
                "later_vs_earlier" -> KL(p_later || p_earlier)"""
    log_p = [tf.nn.log_softmax(s, axis=-1) for s in expert_scores]
    values = []
    for a in range(len(log_p)):
        for b in range(a + 1, len(log_p)):
            if direction == "earlier_vs_later":
                ref, other = log_p[a], log_p[b]
            elif direction == "later_vs_earlier":
                ref, other = log_p[b], log_p[a]
            else:
                raise ValueError("direction must be 'earlier_vs_later' or 'later_vs_earlier'")
            kl = tf.reduce_sum(tf.exp(ref) * (ref - other), axis=-1)       # (batch,)  KL(ref || other)
            values.append(tf.reduce_mean(tf.exp(-kl)))
    return tf.add_n(values) / len(values)


class DiversityPenalty(layers.Layer):
    """Pass-through layer: takes a LIST of expert scores, adds weight * penalty to the loss with
    add_loss, and hands the same scores on unchanged. With fewer than 2 experts it does nothing."""

    def __init__(self, weight=0.1, direction="earlier_vs_later", **kw):
        super().__init__(**kw)
        self.weight, self.direction = weight, direction

    def call(self, expert_scores):
        if len(expert_scores) >= 2:
            self.add_loss(self.weight * diversity_penalty_value(expert_scores, self.direction))
        return [tf.identity(s) for s in expert_scores]


class EnsembleAverage(layers.Layer):
    """Average of the experts' scores. stop_gradient: it is only used to MEASURE the combined
    result (accuracy, val macro-F1); every expert is trained by its own cross-entropy loss."""

    def call(self, expert_scores):
        return tf.stop_gradient(tf.add_n(expert_scores) / len(expert_scores))


def mge_model(inp, expert_scores, active, diversity_weight=0.1, direction="earlier_vs_later"):
    """Wire the experts listed in `active` (e.g. [2, 1, 0]; order = order added) into a Model.
    Outputs: [ensemble, expert_<i>, ...] in that order. All experts of `active` get the
    diversity penalty. Layers are SHARED with any other model made from the same graph."""
    chosen = [expert_scores[i] for i in active]
    chosen = DiversityPenalty(diversity_weight, direction, name="diversity")(chosen)
    outputs = [EnsembleAverage(name="ensemble")(chosen)]
    for i, scores in zip(active, chosen):
        outputs.append(layers.Identity(name=f"expert_{i}")(scores))     # named so history keys are readable
    return tf.keras.Model(inp, outputs)
