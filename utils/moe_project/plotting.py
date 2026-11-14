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
