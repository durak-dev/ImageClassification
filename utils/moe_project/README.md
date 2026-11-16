# moe_project

A small, modular rebuild of `utils/MoE.py` (which is left unchanged) plus two new ideas:
**pMoE** (experts that pick different patches) and **MGE-style experts** (experts at different depths).
The models are used like the prebuilt models of `tf.keras.applications`.

```
config.py     every setting in one place (defaults only; every model class also takes them as arguments)
backbone.py   res_block, backbone (copied from MoE.py) + backbone_feature_map, backbone_multi_depth
moe.py        MixtureOfExperts layer (copied from MoE.py)
pmoe.py       PatchMoE layer + expert_choices() to read back which patches each expert picked
mge.py        depth experts, DiversityPenalty (add_loss), ensemble
models.py     CNNClassifier, MoEClassifier, PMoEClassifier, MGEClassifier + compile_default
callbacks.py  GateMonitor, MacroF1, default_callbacks
train.py      run(model_name, seed, ...) and compare(...): train with the standard callbacks
```
All plotting functions live in `utils/plotting.py` (next to `utils/funcs.py`, which still re-exports `plot_images`
and `plot_history` for older notebooks).

## Use a model like a prebuilt Keras model

```python
from utils.moe_project import PMoEClassifier, compile_default

model = PMoEClassifier(input_shape=(64, 64, 3), classes=200, n_experts=4, l=8)   # like ResNet50(input_shape=..., classes=...)
compile_default(model)              # the project's standard compile (AdamW, cross-entropy on logits), or your own model.compile(...)
model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=10)
```
Each class is a normal `tf.keras.Model`: `summary()`, `fit`, `evaluate`, `predict`, `save_weights` and
`model.save("x.keras")` + `tf.keras.models.load_model("x.keras")` work. Like the prebuilt models, they come back
**not compiled**, and they output **logits**, so use `from_logits=True` in your own loss.
(`save()` does not store the `augmentation` layer; pass it again when you rebuild.)

| class | call | what it is |
|---|---|---|
| `CNNClassifier` | `CNNClassifier(input_shape, classes, n_experts=4, hidden=128)` | baseline; dense head with `n_experts*hidden` units |
| `MoEClassifier` | `MoEClassifier(input_shape, classes, n_experts=4, hidden=128, balance=0.05)` | plain mixture on pooled features |
| `PMoEClassifier` | `PMoEClassifier(input_shape, classes, n_experts=4, l=8, hidden=64)` | patch routing; `l` must be <= `n_patches(size)` |
| `MGEClassifier` | `MGEClassifier(input_shape, classes, diversity_weight=0.1)` | 3 depth experts trained together (version 1) |
| `MGEClassifier(...).stages()` | | a LIST of 3 stage models sharing their layers (version 2, gradual) |

All of them also take `augmentation=` (a layer or `Sequential`, applied first), `drop=` (backbone dropout),
and `expert_drop=` (`head_drop=` for the CNN). Defaults come from `config.py`.

The **MGE model has several outputs**: `[ensemble, expert_2, expert_1, expert_0]` (the order follows `order`).
The ensemble is the average of the experts' scores and is only used for measuring (accuracy, val macro-F1);
each expert is trained by its own cross-entropy. You can still use it like a one-output model: a normal
`model.compile(optimizer, loss, metrics=["accuracy"])` is repeated for every output, and `fit` accepts plain
`(image, label)` datasets. `predict` returns a list with one array per output, and the history has one set of keys per
output (`ensemble_accuracy`, `expert_0_loss`, ...). Its plain `loss` is the sum over all outputs.

How many patches does pMoE have? `n_patches(size)` in `backbone.py`: size 64 -> 64 patches (8x8), size 32 -> 16 (4x4).

## Train with the standard callbacks (notebook)

`train.run` builds, compiles and trains a model with val macro-F1, a CSV log and early stopping, and returns the History.

```python
import sys; sys.path.insert(0, ROOT)           # ROOT = the repo folder (as in MoE_approach.ipynb)
from utils.moe_project.train import run, compare
from utils import plotting

history = run("pmoe", seed=0, train_ds=train_ds, val_ds=val_ds, input_shape=(SIZE, SIZE, 3), classes=N_CLASSES)
plotting.plot_history(history)

compare(train_ds, val_ds, input_shape=(SIZE, SIZE, 3), classes=N_CLASSES)   # cnn, moe, pmoe, mge x seeds in config.SEEDS
```
Model names for `run`: `"cnn"`, `"moe"`, `"pmoe"`, `"mge"`, `"mge_gradual"`. Extra arguments go to the model class:
`run("pmoe", 0, train_ds, val_ds, input_shape=(64, 64, 3), classes=10, n_experts=8, l=4)`. `lr`, `weight_decay`, `epochs`,
`patience`, `log_dir` are arguments of `run` itself. `compare` prints one row per model: parameter count and best val
macro-F1 (mean and per seed). Set `IMG_SIZE` and `N_CLASSES` in `config.py` once to leave `input_shape=`/`classes=` out.

## Plots (all in `utils/plotting.py`)

```python
from utils import plotting
plotting.plot_history(history)                      # loss + val macro-F1 of one run
plotting.plot_comparison({"cnn": hist_cnn, "pmoe": hist_pmoe})   # several runs (History, dict or CSV DataFrame)
plotting.plot_images(df, path_col="path", n=16)     # a grid of images from a dataframe
```
Which patches did each pMoE expert pick?
```python
from utils.moe_project.pmoe import expert_choices
images, labels = next(iter(val_ds))
top_idx, gates, grid = expert_choices(history.model, images[:4])     # top_idx: (4 images, k experts, l patches)
plotting.show_expert_patches(images[0].numpy(), top_idx[0], grid)    # one panel per expert
```

## Common edits: which file, which argument

| I want to... | change this |
|---|---|
| more / fewer experts (cnn, moe, pmoe) | `config.py` `N_EXPERTS`, or `n_experts=` in the class / `run(...)` |
| bigger hidden layer, cnn and moe | `config.py` `HIDDEN`, or `hidden=` |
| bigger hidden layer, pMoE experts | `config.py` `PMOE_HIDDEN`, or `hidden=` in `PMoEClassifier` |
| different `l` (patches kept per expert) | `config.py` `L_PATCHES`, or `l=` in `PMoEClassifier` (the paper's name for it) |
| different diversity weight | `config.py` `DIVERSITY_WEIGHT`, or `diversity_weight=` in `MGEClassifier` |
| flip the KL direction | `config.py` `DIVERSITY_DIRECTION` / `direction=` |
| which depth expert comes first (and so "earlier" in the penalty / gradual schedule) | `config.py` `MGE_ORDER` / `order=` |
| epochs per gradual stage | `config.py` `MGE_STAGE_EPOCHS` / `stage_epochs=` in `run` |
| more dropout in experts | `EXPERT_DROP` / `expert_drop=` |
| load-balancing strength (moe only) | `BALANCE` / `balance=` |
| learning rate, weight decay | `LR`, `WEIGHT_DECAY`, or `compile_default(model, lr=..., weight_decay=...)` / `run(lr=...)` |
| epochs, patience, seeds | `EPOCHS`, `PATIENCE`, `SEEDS` |
| change what the backbone looks like | `backbone.py` (all four models read from it) |
| change how patches are scored or gated | `pmoe.py`, method `PatchMoE.route` |
| change the diversity formula | `mge.py`, function `diversity_penalty_value` |
| add a 4th MGE expert at another depth | `backbone.py` `backbone_multi_depth` (return one more map), `mge.py` `mge_graph` picks it up; extend `MGE_ORDER` and the `range(3)` in `MGEClassifier.stages` |

## What differs from MoE.py (and what is a guess)

Unchanged behaviour: `res_block`, `backbone`, `MixtureOfExperts`, `GateMonitor`.
Small changes you should know about:
* The models are Keras model classes now (`CNNClassifier`, `MoEClassifier`, ...) and are returned **uncompiled**. `compile_default`
  is the old `_compile` (same optimizer, loss and metric).
* `MoEClassifier(..., hidden=128)`: `hidden` is now an argument (the old default is kept).
* `MacroF1(val_ds, output_index=0)`: picks the ensemble output of multi-output models; no change for the others.

**pMoE** (Chowdhury et al., ICML 2023, arXiv 2306.04073). I read the paper's model section (3.1-3.2): per-expert router
vector, top-l patches, softmax gates over the kept scores, expert outputs added with a 1/l factor. Things to know:
the arXiv text prints the softmax denominator without `exp` (I used a normal softmax); the paper's theory
is for a two-layer CNN on disjoint patches with a binary label, so using it on a ResNet feature map with many classes
is an **adaptation**; I only saw a machine summary of the paper's experiments section, not the text itself.

**MGE** (Zhang et al., ICCV 2019). **I could not read this paper** (the official PDF is blocked for automated download).
So these parts are *not* from the paper:
* the diversity formula: `exp(-KL(p_earlier || p_later))`, averaged over all pairs. It is modelled on a community
  re-implementation (github.com/kiminh/MGE-CNN), not on the paper, and that code also removes the true class before the KL
  (needs labels, impossible in `add_loss`);
* the direction and the weight (0.1 is just a starting value);
* the gradual schedule (version 2): only "add experts one at a time" comes from the paper's abstract. The order
  (deepest first), the fact that old experts keep training, and the epochs per stage are my choices;
* "granularity by backbone depth" is your design; the paper's granularities are image scales (whole image, crops).

Please check `diversity_penalty_value` in `mge.py` against the paper's equation before relying on MGE results.
