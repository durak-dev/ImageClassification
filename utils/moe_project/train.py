"""Train any of the models with one call, and compare them side by side.

    from utils.moe_project.train import run, compare
    history = run("pmoe", seed=0, train_ds=train_ds, val_ds=val_ds, input_shape=(64, 64, 3), classes=10)
    compare(train_ds, val_ds, input_shape=(64, 64, 3), classes=10)

train_ds / val_ds are tf.data datasets of (image, label) batches, as in MoE.py. val_ds must NOT be shuffled.
(If you build a model yourself, e.g. PMoEClassifier(...), you do not need this file: compile and fit it directly.)
"""
import numpy as np
import tensorflow as tf

from . import config
from .callbacks import default_callbacks
from .models import (CNNClassifier, MoEClassifier, PMoEClassifier, MGEClassifier, compile_default,
                     DEFAULT_INPUT_SHAPE)

MODEL_CLASSES = {"cnn": CNNClassifier, "moe": MoEClassifier, "pmoe": PMoEClassifier, "mge": MGEClassifier}
ALL_NAMES = ("cnn", "moe", "pmoe", "mge", "mge_gradual")


def _fit(model, train_ds, val_ds, epochs, name, patience, log_dir, gate=False):
    """One fit() call with the standard callbacks."""
    cbs, _ = default_callbacks(val_ds, name=name, log_dir=log_dir, patience=patience, gate=gate)
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2)


def run(model_name, seed, train_ds, val_ds, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES,
        augmentation=None, epochs=config.EPOCHS, patience=config.PATIENCE, log_dir=config.LOG_DIR,
        lr=config.LR, weight_decay=config.WEIGHT_DECAY, stage_epochs=config.MGE_STAGE_EPOCHS, **model_args):
    """Build, compile and train ONE model, return the Keras History (history.history has val_f1;
    history.model is the trained model).
    model_name : "cnn", "moe", "pmoe", "mge" or "mge_gradual"
    model_args : anything else goes to the model class, e.g. run("pmoe", 0, ..., n_experts=8, l=4)
    "mge_gradual" trains stage by stage: every stage but the last for `stage_epochs`, the last for `epochs`."""
    if model_name not in ALL_NAMES:
        raise ValueError(f"model_name must be one of {ALL_NAMES}, got {model_name!r}")
    tf.keras.backend.clear_session()                 # forget layers of earlier runs (also frees memory)
    tf.keras.utils.set_random_seed(seed)             # same seed -> same start weights and batch order

    run_name = f"{model_name}_seed{seed}"

    if model_name != "mge_gradual":
        model = MODEL_CLASSES[model_name](input_shape=input_shape, classes=classes,
                                          augmentation=augmentation, **model_args)
        compile_default(model, lr, weight_decay)
        return _fit(model, train_ds, val_ds, epochs, run_name, patience, log_dir, gate=(model_name == "moe"))

    # ---- gradual MGE: one stage after the other, the stage models share their layers
    stages = MGEClassifier(input_shape=input_shape, classes=classes, augmentation=augmentation,
                           **model_args).stages()
    merged = tf.keras.callbacks.History()            # one History that collects every stage
    merged.history = {"stage": []}
    for number, stage_model in enumerate(stages, start=1):
        last = number == len(stages)
        print(f"--- stage {number}/{len(stages)}: {len(stage_model.outputs) - 1} expert(s)")
        compile_default(stage_model, lr, weight_decay)
        stage_history = _fit(stage_model, train_ds, val_ds, epochs if last else stage_epochs,
                             f"{run_name}_stage{number}", patience, log_dir)
        for key in ("loss", "val_loss", "val_f1", "ensemble_loss", "val_ensemble_loss",
                    "ensemble_accuracy", "val_ensemble_accuracy"):
            merged.history.setdefault(key, []).extend(stage_history.history[key])
        merged.history["stage"].extend([number] * len(stage_history.history["loss"]))
    merged.set_model(stages[-1])                     # history.model = the model with all experts
    return merged


def compare(train_ds, val_ds, input_shape=DEFAULT_INPUT_SHAPE, classes=config.N_CLASSES, seeds=config.SEEDS,
            model_names=("cnn", "moe", "pmoe", "mge"), **run_args):
    """Train every model in model_names on the SAME data and seeds, then print parameter counts
    next to the best val macro-F1 (best epoch of each run; mean over seeds). Returns the table as a dict.
    run_args are passed to run(), e.g. epochs=5 for a quick test."""
    results = {}
    for name in model_names:
        f1_per_seed, params = [], None
        for seed in seeds:
            print(f"\n===== {name}  seed {seed}")
            history = run(name, seed, train_ds, val_ds, input_shape=input_shape, classes=classes, **run_args)
            f1_per_seed.append(max(history.history["val_f1"]))
            params = history.model.count_params()
        results[name] = {"params": params, "best_val_f1": f1_per_seed}

    print("\n" + "model".ljust(12) + "params".rjust(12) + "best val macro-F1 (mean)".rjust(28) + "   per seed")
    for name, r in results.items():
        per_seed = "  ".join(f"{f:.4f}" for f in r["best_val_f1"])
        print(name.ljust(12) + f"{r['params']:>12,}" + f"{np.mean(r['best_val_f1']):>28.4f}" + "   " + per_seed)
    return results
