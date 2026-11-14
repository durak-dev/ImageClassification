"""One place for every setting of the MoE project.

Nothing else in this folder reads this file on its own: only models.py and
train.py import it, and they use these values as DEFAULT arguments. So every
builder can still be called with different values, e.g.
    build_pmoe(size, n_classes, n_experts=8, l=4)
without touching this file.

(Note: this is NOT utils/config.py, which holds the GitHub settings.)
"""

# ---------------------------------------------------------------- data (fill in)
IMG_SIZE = 64          # images are IMG_SIZE x IMG_SIZE x 3.  <-- set this to your size
N_CLASSES = 200         # number of classes.                    <-- set this to your number

# ---------------------------------------------------------------- shared backbone
DROP = 0.3             # spatial dropout inside the backbone (feature dropout = DROP * 2.5)

# ---------------------------------------------------------------- shared by cnn / moe / pmoe
N_EXPERTS = 8          # number of experts (cnn uses it only to size its dense head)
HIDDEN = 128           # hidden units of ONE expert
EXPERT_DROP = 0.3      # dropout inside the experts / dense head

# ---------------------------------------------------------------- moe (MoE.py)
BALANCE = 0.05         # weight of the load-balancing loss

# ---------------------------------------------------------------- pmoe (patch routing)
L_PATCHES = 8          # l = how many patches each expert keeps (must be <= number of patches)
PMOE_HIDDEN = 64       # hidden units of one pMoE expert (it runs on every kept patch)

# ---------------------------------------------------------------- mge (granularity experts)
DIVERSITY_WEIGHT = 0.1     # weight of the KL diversity penalty
DIVERSITY_DIRECTION = "earlier_vs_later"   # or "later_vs_earlier"  (see mge.DiversityPenalty)
MGE_ORDER = (2, 1, 0)      # order experts are used in: 0 = 64 ch, 1 = 128 ch, 2 = 256 ch (deepest)
MGE_STAGE_EPOCHS = 5       # gradual version only: epochs for each stage before the last one

# ---------------------------------------------------------------- training
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 40
PATIENCE = 8           # early stopping patience (epochs)
SEEDS = (0, 1, 2)      # the seeds compare() loops over
LOG_DIR = "history_logs"   # per-epoch history CSVs are written here
