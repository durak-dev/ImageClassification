# moe_project

A small, modular rebuild of `utils/MoE.py` (which is left unchanged) plus two new ideas:
**pMoE** (experts that pick different patches) and **MGE-style experts** (experts at different depths).

```
config.py     every setting in one place (defaults only; every builder also takes them as arguments)
backbone.py   res_block, backbone (copied from MoE.py) + backbone_feature_map, backbone_multi_depth
moe.py        MixtureOfExperts (copied from MoE.py)
pmoe.py       PatchMoE layer + expert_choices() to read back which patches each expert picked
mge.py        depth experts, DiversityPenalty (add_loss), ensemble
models.py     build_cnn, build_moe, build_pmoe, build_mge, build_mge_gradual (+ _compile)
callbacks.py  GateMonitor, MacroF1, default_callbacks
plotting.py   plot_history, plot_comparison (several runs in one figure), show_expert_patches
train.py      run(model_name, seed, ...) and compare(...)
```

## Quick start (notebook)

```python
import sys; sys.path.insert(0, ROOT)           # ROOT = the repo folder (as in MoE_approach.ipynb)
from utils.moe_project.train import run, compare
from utils.moe_project.plotting import plot_history

history = run("pmoe", seed=0, train_ds=train_ds, val_ds=val_ds, size=SIZE, n_classes=N_CLASSES)
plot_history(history)

compare(train_ds, val_ds, size=SIZE, n_classes=N_CLASSES)     # cnn, moe, pmoe, mge x all seeds in config.SEEDS
```
`compare` prints one row per model: parameter count and best val macro-F1 (mean and per seed).
Set `IMG_SIZE` and `N_CLASSES` in `config.py` once, then you can leave `size=`/`n_classes=` out.

## Calling each builder

All take `(size, n_classes, ...)` and return a compiled model (same AdamW, same loss).

| builder | call | notes |
|---|---|---|
| `build_cnn` | `build_cnn(size, n_classes, n_experts=4, hidden=128)` | baseline; dense head with `n_experts*hidden` units |
| `build_moe` | `build_moe(size, n_classes, n_experts=4, hidden=128, balance=0.05)` | plain mixture on pooled features |
| `build_pmoe` | `build_pmoe(size, n_classes, n_experts=4, l=8, hidden=64)` | patch routing; `l` must be <= `n_patches(size)` |
| `build_mge` | `build_mge(size, n_classes, diversity_weight=0.1)` | 3 depth experts trained together (version 1) |
| `build_mge_gradual` | `build_mge_gradual(size, n_classes)` | returns a LIST of 3 stage models (version 2) |

Through `run` the extra arguments go to the builder: `run("pmoe", 0, train_ds, val_ds, size=64, n_classes=10, n_experts=8, l=4)`.
The model names for `run` are `"cnn"`, `"moe"`, `"pmoe"`, `"mge"`, `"mge_gradual"`.

The **MGE model has several outputs**: `[ensemble, expert_2, expert_1, expert_0]` (the order follows `order`).
The ensemble is the average of the experts' scores and is only used for measuring (accuracy, val macro-F1);
each expert is trained by its own cross-entropy. `train.run` takes care of giving every output the label.

How many patches does pMoE have? `n_patches(size)` in `backbone.py`: size 64 -> 64 patches (8x8), size 32 -> 16 (4x4).

Which patches did each pMoE expert pick?
```python
from utils.moe_project.pmoe import expert_choices
from utils.moe_project.plotting import show_expert_patches
images, labels = next(iter(val_ds))
top_idx, gates, grid = expert_choices(history.model, images[:4])     # top_idx: (4 images, k experts, l patches)
show_expert_patches(images[0].numpy(), top_idx[0], grid)             # one panel per expert
```

## Common edits: which file, which argument

| I want to... | change this |
|---|---|
| more / fewer experts (cnn, moe, pmoe) | `config.py` `N_EXPERTS`, or `n_experts=` in the builder / `run(...)` |
| bigger hidden layer, cnn and moe | `config.py` `HIDDEN`, or `hidden=` |
| bigger hidden layer, pMoE experts | `config.py` `PMOE_HIDDEN`, or `hidden=` in `build_pmoe` |
| different `l` (patches kept per expert) | `config.py` `L_PATCHES`, or `l=` in `build_pmoe` (the paper's name for it) |
| different diversity weight | `config.py` `DIVERSITY_WEIGHT`, or `diversity_weight=` in `build_mge` |
| flip the KL direction | `config.py` `DIVERSITY_DIRECTION` / `direction=` |
| which depth expert comes first (and so "earlier" in the penalty / gradual schedule) | `config.py` `MGE_ORDER` / `order=` |
| epochs per gradual stage | `config.py` `MGE_STAGE_EPOCHS` / `stage_epochs=` in `run` |
| more dropout in experts | `EXPERT_DROP` / `expert_drop=` |
| load-balancing strength (moe only) | `BALANCE` / `balance=` |
| learning rate, epochs, patience, seeds | `LR`, `EPOCHS`, `PATIENCE`, `SEEDS` |
| change what the backbone looks like | `backbone.py` (all four models read from it) |
| change how patches are scored or gated | `pmoe.py`, method `PatchMoE.route` |
| change the diversity formula | `mge.py`, function `diversity_penalty_value` |
| add a 4th MGE expert at another depth | `backbone.py` `backbone_multi_depth` (return one more map), then `mge.py` `mge_graph` picks it up; extend `MGE_ORDER` |

## What differs from MoE.py (and what is a guess)

Unchanged behaviour: `res_block`, `backbone`, `MixtureOfExperts`, `GateMonitor`, `build_cnn`, `build_moe`.
Small additions you should know about:
* `_compile(..., n_outputs=1)`: with one output it is exactly the old function; the MGE model needs the loss repeated per output.
* `MacroF1(val_ds, output_index=0)`: picks the ensemble output of multi-output models; no change for the others.
* `build_moe(..., hidden=128)`: `hidden` is now an argument (the old default is kept).

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
