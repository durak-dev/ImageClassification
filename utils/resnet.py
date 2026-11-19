"""File helpers for the saved backbones (MoE_comparisons.ipynb, pMoE_MGE_finetune.ipynb).

A trained backbone ("trunk") is saved as a complete Keras model (`<name>.keras`: layers AND weights) with
`trunk.save(path)`, so a notebook that loads it with `tf.keras.models.load_model` does not need the code that built it.
"""
import os
import time

import tensorflow as tf


def norm_path(path):
    """Accept a Windows-style path (backslashes) on Linux/Colab, and a path inside the project folder.
    The result uses '/' and is relative to the project folder, which is also how the file is named on GitHub."""
    p = str(path).replace(chr(92), '/')
    root = os.getcwd().replace(chr(92), '/').rstrip('/') + '/'
    return p[len(root):] if p.startswith(root) else p


def new_trunk_path(models_dir, trunk):
    """Automatic, unique file name for a new backbone: <models_dir>/backbone_<params>k_<date-time>[_fp16].keras
    (_fp16 when the notebook runs with mixed_float16: the layers are then saved in fp16 and load that way)."""
    tag = '_fp16' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else ''
    return f'{models_dir}/backbone_{trunk.count_params() // 1000}k_{time.strftime("%Y%m%d-%H%M%S")}{tag}.keras'


def fetch_if_missing(path, use_github=False):
    """True if `path` exists afterwards. With use_github a missing file is first downloaded from the
    checkpoints branch (a new pod / Colab VM has an empty disk)."""
    path = norm_path(path)
    if not os.path.exists(path) and use_github:
        from utils import github_sync
        try:
            content = github_sync.fetch_file(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            print(f'fetched {path} from GitHub')
        except FileNotFoundError:
            print(f'{path} is not on GitHub either (check the path and the branch)')
        except Exception as e:
            print(f'could not fetch {path} from GitHub ({e})')
    if not os.path.exists(path) and not use_github:
        print(f'{path} is not on this disk and GitHub is off (no token): it cannot be downloaded')
    return os.path.exists(path)
