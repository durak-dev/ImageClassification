"""Training callbacks, copied from MoE.py.

Differences from MoE.py (everything else is identical):
  * MacroF1 got an extra argument output_index (default 0). Models with ONE output
    behave exactly as before. The MGE model has several outputs and predict() then returns
    a list; output_index says which one to score (0 = the "ensemble" output).
  * default_callbacks() is here too, because train.py needs it.
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score


class GateMonitor(tf.keras.callbacks.Callback):
    """Prints how much each expert of the plain MoE (layer name "moe") is used. MoE model only."""
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
    """Computes macro F1 on val_ds at the end of each epoch; stores it in logs['val_f1'].
    output_index: which model output to score when the model has several (see top of file)."""
    def __init__(self, val_ds, output_index=0):
        super().__init__()
        self.val_ds = val_ds
        self.output_index = output_index
        self.y_true = np.concatenate([y.numpy() for _, y in val_ds])   # labels fixed, val not shuffled

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.val_ds, verbose=0)
        if isinstance(pred, (list, tuple)):          # multi-output model (MGE)
            pred = pred[self.output_index]
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
