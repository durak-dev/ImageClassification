"""Train any of the models with one call, and compare them side by side.

    from utils.moe_project.train import run, compare
    history = run("pmoe", seed=0, train_ds=train_ds, val_ds=val_ds, size=64, n_classes=10)
    compare(train_ds, val_ds, size=64, n_classes=10)

train_ds / val_ds are tf.data datasets of (image, label) batches, as in MoE.py. val_ds must NOT be shuffled.
"""
import numpy as np
import tensorflow as tf

from . import config
from .callbacks import default_callbacks
from .models import build_cnn, build_moe, build_pmoe, build_mge, build_mge_gradual

BUILDERS = {"cnn": build_cnn, "moe": build_moe, "pmoe": build_pmoe, "mge": build_mge}
ALL_NAMES = ("cnn", "moe", "pmoe", "mge", "mge_gradual")


def _label_per_output(ds, n_outputs):
    """The MGE model has one output per expert (+ the ensemble) and each gets the same label,
    so every (image, label) becomes (image, (label, label, ...))."""
    return ds.map(lambda x, y: (x, tuple([y] * n_outputs)))


def _fit(model, train_ds, val_ds, epochs, name, patience, log_dir, gate=False):
    """One fit() call with the standard callbacks."""
    n_outputs = len(model.outputs)
    cbs, _ = default_callbacks(val_ds, name=name, log_dir=log_dir, patience=patience, gate=gate)
    if n_outputs > 1:                                    # MGE: labels once per output
        fit_train, fit_val = _label_per_output(train_ds, n_outputs), _label_per_output(val_ds, n_outputs)
    else:
        fit_train, fit_val = train_ds, val_ds
    return model.fit(fit_train, validation_data=fit_val, epochs=epochs, callbacks=cbs, verbose=2)


def run(model_name, seed, train_ds, val_ds, size=config.IMG_SIZE, n_classes=config.N_CLASSES,
        augmentation=None, epochs=config.EPOCHS, patience=config.PATIENCE,
        log_dir=config.LOG_DIR, stage_epochs=config.MGE_STAGE_EPOCHS, **builder_args):
    """Build and train ONE model, return the Keras History (history.history has val_f1;
    history.model is the trained model).
    model_name : "cnn", "moe", "pmoe", "mge" or "mge_gradual"
    builder_args : anything else is passed to the builder, e.g. run("pmoe", 0, ..., n_experts=8, l=4)
    "mge_gradual" trains stage by stage: every stage but the last for `stage_epochs`, the last for `epochs`."""
    if model_name not in ALL_NAMES:
        raise ValueError(f"model_name must be one of {ALL_NAMES}, got {model_name!r}")
    tf.keras.backend.clear_session()                 # forget layers of earlier runs (also frees memory)
    tf.keras.utils.set_random_seed(seed)             # same seed -> same start weights and batch order

    run_name = f"{model_name}_seed{seed}"

    if model_name != "mge_gradual":
        model = BUILDERS[model_name](size, n_classes, augmentation=augmentation, **builder_args)
        return _fit(model, train_ds, val_ds, epochs, run_name, patience, log_dir, gate=(model_name == "moe"))

    # ---- gradual MGE: one stage after the other, models share their weights
    stages = build_mge_gradual(size, n_classes, augmentation=augmentation, **builder_args)
    merged = tf.keras.callbacks.History()            # one History that collects every stage
    merged.history = {"stage": []}
    for number, model in enumerate(stages, start=1):
        last = number == len(stages)
        print(f"--- stage {number}/{len(stages)}: {len(model.outputs) - 1} expert(s)")
        stage_history = _fit(model, train_ds, val_ds, epochs if last else stage_epochs,
                             f"{run_name}_stage{number}", patience, log_dir)
        for key in ("loss", "val_loss", "val_f1", "ensemble_loss", "val_ensemble_loss",
                    "ensemble_accuracy", "val_ensemble_accuracy"):
            merged.history.setdefault(key, []).extend(stage_history.history[key])
        merged.history["stage"].extend([number] * len(stage_history.history["loss"]))
    merged.set_model(stages[-1])                     # history.model = the model with all experts
    return merged


def compare(train_ds, val_ds, size=config.IMG_SIZE, n_classes=config.N_CLASSES, seeds=config.SEEDS,
            model_names=("cnn", "moe", "pmoe", "mge"), **run_args):
    """Train every model in model_names on the SAME data and seeds, then print parameter counts
    next to the best val macro-F1 (best epoch of each run; mean over seeds). Returns the table as a dict.
    run_args are passed to run(), e.g. epochs=5 for a quick test."""
    results = {}
    for name in model_names:
        f1_per_seed, params = [], None
        for seed in seeds:
            print(f"\n===== {name}  seed {seed}")
            history = run(name, seed, train_ds, val_ds, size=size, n_classes=n_classes, **run_args)
            f1_per_seed.append(max(history.history["val_f1"]))
            params = history.model.count_params()
        results[name] = {"params": params, "best_val_f1": f1_per_seed}

    print("\n" + "model".ljust(12) + "params".rjust(12) + "best val macro-F1 (mean)".rjust(28) + "   per seed")
    for name, r in results.items():
        per_seed = "  ".join(f"{f:.4f}" for f in r["best_val_f1"])
        print(name.ljust(12) + f"{r['params']:>12,}" + f"{np.mean(r['best_val_f1']):>28.4f}" + "   " + per_seed)
    return results
