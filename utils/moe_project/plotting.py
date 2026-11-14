"""Plots: the training history, and which patches each pMoE expert picked."""
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history):
    """Loss curves and val macro-F1 (same as plot_history in utils/funcs.py).
    Works for every model: it only uses the keys loss, val_loss and val_f1."""
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    if "val_f1" in h:
        ax[1].plot(h["val_f1"], color="C2", marker="o")
        best = int(np.argmax(h["val_f1"]))
        ax[1].set(title=f"Val macro-F1 (best {h['val_f1'][best]:.3f} @ epoch {best})", xlabel="epoch")
    plt.tight_layout(); plt.show()


def _as_dict(run):
    """A run can be a Keras History, a plain history dict, or a DataFrame (a CSV log read with pandas)."""
    if hasattr(run, "history") and isinstance(run.history, dict):
        return run.history
    if hasattr(run, "to_dict"):
        return run.to_dict("list")
    return run


def _first_key(h, *names):
    """The first of these names that exists in the history."""
    for name in names:
        if name in h:
            return name
    raise KeyError(f"none of {names} found; the history has {list(h)}")


def plot_comparison(runs):
    """Compare several runs (cnn, moe, pmoe, mge, ...) in one figure: loss, val macro-F1, val accuracy.
    runs: dict  name -> run,  where a run is a Keras History (what train.run returns),
          a history dict, or a DataFrame read from a CSV log, e.g.
          runs = {"cnn": hist_cnn, "moe": pd.read_csv("history_logs/moe_seed0.csv")}

    The MGE model has several outputs, and its plain 'loss' is the SUM over all outputs, which cannot be
    compared with the other models. For it the figure uses the loss and accuracy of its 'ensemble' output
    (the average of the experts) instead, which are comparable. Also note that the MoE loss includes its
    small balance penalty. For a gradual MGE run, dotted vertical lines mark where a new stage starts."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for i, (name, run) in enumerate(runs.items()):
        h = _as_dict(run)
        color = f"C{i}"

        train_loss = h[_first_key(h, "ensemble_loss", "loss")]
        val_loss = h[_first_key(h, "val_ensemble_loss", "val_loss")]
        val_acc = h[_first_key(h, "val_ensemble_accuracy", "val_accuracy")]
        val_f1 = h["val_f1"]

        label_f1 = f"{name} (best {max(val_f1):.3f})"
        try:                                              # parameter count, only known for a History
            label_f1 = f"{name} ({run.model.count_params() / 1e6:.2f}M params, best {max(val_f1):.3f})"
        except Exception:
            pass

        ax[0].plot(val_loss, color=color, label=name)
        ax[0].plot(train_loss, color=color, ls="--", alpha=0.5)          # dashed = train loss
        ax[1].plot(val_f1, color=color, label=label_f1)
        ax[2].plot(val_acc, color=color, label=f"{name} (best {max(val_acc):.3f})")

        if "stage" in h:                                  # gradual MGE: mark the start of each new stage
            stage = list(h["stage"])
            for epoch in range(1, len(stage)):
                if stage[epoch] != stage[epoch - 1]:
                    for a in ax:
                        a.axvline(epoch - 0.5, color=color, ls=":", alpha=0.6)

    ax[0].set(title="Loss (solid = val, dashed = train)", xlabel="epoch")
    ax[1].set(title="Val macro-F1", xlabel="epoch")
    ax[2].set(title="Val accuracy", xlabel="epoch")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def show_expert_patches(image, top_idx, grid, expert=None, ax=None):
    """Draw which patches the experts picked for ONE image.
    image   : (size, size, 3) array
    top_idx : (k, l) patch numbers for this image  (= expert_choices(...)[0][i])
    grid    : (H, W) patch grid                    (= expert_choices(...)[2])
    expert  : None -> one panel per expert; or one expert number -> a single panel
    Each picked patch is outlined in the expert's colour (patch p is at row p // W, column p % W)."""
    size_h, size_w = image.shape[0], image.shape[1]
    grid_h, grid_w = grid
    cell_h, cell_w = size_h / grid_h, size_w / grid_w
    experts = range(top_idx.shape[0]) if expert is None else [expert]

    if ax is None:
        fig, axes = plt.subplots(1, len(experts), figsize=(3.2 * len(experts), 3.4), squeeze=False)
        axes = axes[0]
    else:
        axes = [ax]

    for a, e in zip(axes, experts):
        a.imshow(np.clip(image / 255.0 if image.max() > 1.5 else image, 0, 1))
        for p in top_idx[e]:
            row, col = divmod(int(p), grid_w)
            a.add_patch(plt.Rectangle((col * cell_w, row * cell_h), cell_w, cell_h,
                                      fill=False, edgecolor=f"C{e}", linewidth=2))
        a.set_title(f"expert {e}"); a.axis("off")
    if ax is None:
        plt.tight_layout(); plt.show()
