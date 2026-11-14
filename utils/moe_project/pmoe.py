"""pMoE: patch-level routing (Chowdhury et al., ICML 2023, arXiv 2306.04073).

The idea, in plain words
  * The conv backbone gives a feature map of shape (H, W, C). Every spatial position is one "patch",
    so there are n = H * W patches, each a vector of C numbers.
  * There are k experts. EACH expert has its OWN router: one trainable vector of length C.
    The router scores every patch with a dot product:  score = patch . router_vector
  * An expert only looks at the l best-scoring patches (tf.math.top_k). Different experts
    can pick different patches - that is the whole point.
  * The gate value of a kept patch is the softmax of the kept scores (the paper's "joint training").
  * An expert is a small network on each kept patch. Its output is the gate-weighted
    sum over the l patches (divided by l, as in the paper), turned into class scores.
  * The k expert outputs are ADDED (the paper has no averaging over experts).

How this follows the paper (Eq. 1 and Eq. 4):
    f(x) = sum over experts s of  (1/l) * sum over kept patches j of  expert_s(patch_j) * gate_j,s
    gate_j,s = softmax over the kept patches of  <router_s, patch_j>
  The paper's expert is  sum_r a_r * ReLU(<w_r, patch>)  (a hidden layer and a fixed scalar output weight).
  Here: Dense(hidden, relu) on each patch, gate-weighted pooling, then ONE Dense(n_classes) after the pooling.
  Because pooling is a linear sum, "Dense after the pooling" equals "Dense on every patch, then pool",
  so this is the same function, just cheaper.

THIS IS AN ADAPTATION, NOT A REPRODUCTION. The paper's theory is for a two-layer CNN on
  disjoint image patches with a binary label (logistic loss), where the output weights a_r are
  fixed random. Here the "patches" are positions of a ResNet feature map (they overlap in the
  image through the receptive fields), the output is multi-class, and the output layer is trained.
  The paper's real-data experiment also swaps the last conv layer of a wide ResNet for a pMoE layer,
  so the idea of using it on a feature map is the paper's own, but the guarantees do not carry over.

Two small notes on the paper's formula:
  * Eq. 4 in the arXiv text writes the softmax denominator without the exp(); I use the
    ordinary softmax, as you asked and as the text calls it a softmax.
  * The paper has no load-balancing loss for pMoE, so none is added here. Nothing stops two
    experts from picking the same patches - use expert_choices() to look.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class PatchMoE(layers.Layer):
    """Input: feature map (batch, H, W, C).  Output: class scores (batch, n_classes)."""

    def __init__(self, n_experts, n_classes, l, hidden=64, drop=0.0, **kw):
        super().__init__(**kw)
        self.n_experts, self.n_classes, self.l, self.hidden, self.drop = n_experts, n_classes, l, hidden, drop

    def build(self, input_shape):
        _, h, w, c = input_shape
        self.grid = (h, w)                      # the patch grid, e.g. (8, 8)
        self.n_patches = h * w                  # n = H * W
        self.channels = c
        if self.l > self.n_patches:
            raise ValueError(f"l={self.l} patches per expert, but the feature map only has "
                             f"{self.n_patches} patches ({h}x{w}). Use a smaller l or a bigger image.")

        # one router vector per expert: shape (experts, C). Small start values so all patches
        # begin with similar scores and the softmax starts close to uniform.
        self.router = self.add_weight(name="router", shape=(self.n_experts, c),
                                      initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

        # each expert: hidden layer (run on every kept patch) -> dropout -> output layer (after pooling)
        self.hidden_layers = [layers.Dense(self.hidden, activation="relu", name=f"expert_{i}_hidden")
                              for i in range(self.n_experts)]
        self.drop_layers = [layers.Dropout(self.drop) for _ in range(self.n_experts)]
        self.out_layers = [layers.Dense(self.n_classes, name=f"expert_{i}_out")
                           for i in range(self.n_experts)]
        for i in range(self.n_experts):
            self.hidden_layers[i].build((None, self.l, c))
            self.out_layers[i].build((None, self.hidden))
        super().build(input_shape)

    def route(self, feature_map):
        """Step 1 and 2 of the idea. Returns the patches, which patches each expert kept, and the gates.
            patches : (batch, n, C)   every patch of every image
            top_idx : (batch, k, l)   index of the l patches kept by each expert (0 .. n-1, row by row)
            gates   : (batch, k, l)   softmax of the kept scores (sums to 1 over the l patches)"""
        batch = tf.shape(feature_map)[0]
        patches = tf.reshape(feature_map, (batch, self.n_patches, self.channels))
        scores = tf.einsum("bnc,kc->bkn", patches, self.router)       # (batch, experts, n)
        top_scores, top_idx = tf.math.top_k(scores, k=self.l)         # best l patches PER expert
        gates = tf.nn.softmax(top_scores, axis=-1)                    # softmax over the kept scores only
        return patches, top_idx, gates

    def call(self, feature_map, training=None):
        patches, top_idx, gates = self.route(feature_map)
        chosen = tf.gather(patches, top_idx, batch_dims=1)            # (batch, experts, l, C)

        expert_scores = []
        for i in range(self.n_experts):
            h = self.hidden_layers[i](chosen[:, i])                   # (batch, l, hidden)
            h = self.drop_layers[i](h, training=training)
            weighted = h * gates[:, i, :, None]                       # gate value of each kept patch
            pooled = tf.reduce_sum(weighted, axis=1) / self.l         # (batch, hidden)  the paper's 1/l
            expert_scores.append(self.out_layers[i](pooled))          # (batch, classes)
        return tf.add_n(expert_scores)                                # experts are ADDED, as in the paper


def expert_choices(model, images, layer_name="pmoe"):
    """Read back which patches each expert picked.
    images: a batch of images, shape (batch, size, size, 3) (a numpy array or tensor).
    Returns (top_idx, gates, grid):
        top_idx : numpy (batch, k, l)  patch numbers, counted row by row on the patch grid
        gates   : numpy (batch, k, l)  the gate value of each kept patch
        grid    : (H, W) of the patch grid, so patch p sits at row p // W, column p % W
    Works on any model built with build_pmoe (the layer is called "pmoe")."""
    layer = model.get_layer(layer_name)
    to_feature_map = tf.keras.Model(model.input, layer.input)         # image -> feature map
    feature_map = to_feature_map.predict(images, verbose=0)
    _, top_idx, gates = layer.route(feature_map)
    return top_idx.numpy(), gates.numpy(), layer.grid
