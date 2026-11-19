"""Helpers for backbones: `fn(inp) -> [s1, s2, s3]` is the backbone function written in a notebook.

NEW (MoE_comparisons.ipynb, pMoE_MGE_finetune_v2.ipynb): ONE saved, frozen backbone that every head is trained on.
A backbone is saved as a complete Keras model (`<name>.keras`: layers AND weights), so a notebook that loads it does not
need the code that built it. You choose a backbone by writing its path.

LEGACY (the first versions of the notebooks, MoE_approach.ipynb / pMoE_MGE_finetune.ipynb): weights-only files whose
name carries a fingerprint of the backbone code. Kept unchanged so those notebooks still run.
"""
import hashlib
import inspect
import json
import os
import time
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers

from utils.MoE import res_block


# ---------------------------------------------------------------- shared
def make_trunk(size, fn):
    """The backbone function as one Keras model: image (already augmented) -> [s1, s2, s3]."""
    inp = tf.keras.Input((size, size, 3))
    maps = fn(inp)
    assert isinstance(maps, (list, tuple)) and len(maps) == 3, "backbone must return a list of 3 feature maps [s1, s2, s3]"
    return tf.keras.Model(inp, list(maps), name='trunk')


def trunk_name(path):
    """'MoE_comparisons/models/backbone_6072k_20251118-210000.keras' -> 'backbone_6072k_20251118-210000' (used in log names).
    Also handles the legacy 'trunk_x.weights.h5'."""
    name = os.path.basename(path)
    for ext in ('.weights.h5', '.keras'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def fetch_if_missing(path, use_github=False):
    """True if `path` exists afterwards. With use_github a missing file is first downloaded from the
    checkpoints branch (a new pod / Colab VM has an empty disk)."""
    if not os.path.exists(path) and use_github:
        from utils import github_sync
        try:
            content = github_sync.fetch_file(path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            print(f'fetched {path} from GitHub')
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f'could not fetch {path} from GitHub ({e})')
    return os.path.exists(path)


# ---------------------------------------------------------------- NEW: a saved, frozen backbone
def new_trunk_path(models_dir, trunk):
    """Automatic, unique file name for a new backbone: <models_dir>/backbone_<params>k_<date-time>.keras"""
    return f'{models_dir}/backbone_{trunk.count_params() // 1000}k_{time.strftime("%Y%m%d-%H%M%S")}.keras'


def save_trunk(trunk, path):
    """Save layers + weights. Refuses to overwrite: a backbone other runs depend on must not change silently."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists. Choose another name, or delete the file on purpose.')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    trunk.save(path)
    return path


def load_trunk(path):
    """Load a saved backbone and freeze it. Raises an error that says what is wrong."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist (paths are relative to the project folder). '
                                'Write the path of an existing .keras file, or set BACKBONE_PATH = None to train a new backbone.')
    try:
        trunk = tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        raise ValueError(f'{path} could not be loaded as a backbone ({type(e).__name__}: {e}). '
                         'It must be a .keras file saved with resnet.save_trunk.') from e
    if len(trunk.outputs) != 3:
        raise ValueError(f'{path} has {len(trunk.outputs)} outputs; a backbone must return 3 feature maps [s1, s2, s3].')
    trunk.trainable = False
    return trunk


def trunk_size(path):
    """Input image size of a saved backbone."""
    trunk = load_trunk(path)
    size = trunk.input_shape[1]
    tf.keras.backend.clear_session()
    return size


Backbones = namedtuple('Backbones', 'maps last pooled')
Backbones.__doc__ = """maps:   [s1, s2, s3] (MGE)
last:   s3, the 8x8x256 map before pooling (pMoE)
pooled: s3 pooled to a 256-d vector + feature dropout (CNN, MoE)"""


def make_frozen_backbones(path, feature_drop=0.3):
    """Backbone functions `fn(inp, augmentation)` for build_cnn / build_moe / build_pmoe / build_mge.
    The backbone is FROZEN and run in inference mode (batch norm uses its saved statistics, no dropout inside it),
    so every head is trained on exactly the same features and the file is never changed."""
    def maps(inp, augmentation=None):
        x = augmentation(inp) if augmentation is not None else inp
        return list(load_trunk(path)(x, training=False))

    def last(inp, augmentation=None):
        return maps(inp, augmentation)[-1]

    def pooled(inp, augmentation=None):
        x = layers.GlobalAveragePooling2D()(last(inp, augmentation))
        return layers.Dropout(feature_drop)(x)

    return Backbones(maps, last, pooled)


# ---------------------------------------------------------------- LEGACY (MoE_approach.ipynb / pMoE_MGE_finetune.ipynb)
def backbone_id(size, settings, fn):
    """Parameter count + a short hash of the settings and of the code of the backbone (comments ignored).
    It is part of the file name, so a changed backbone never picks up the weights of an old one."""
    try:
        code = inspect.getsource(fn) + inspect.getsource(res_block)
    except (OSError, TypeError):      # source not available: then only the settings are hashed
        code = ''
    code = '\n'.join(l.split('#')[0].rstrip() for l in code.splitlines() if l.split('#')[0].strip())
    digest = hashlib.md5((code + json.dumps(settings, sort_keys=True)).encode()).hexdigest()[:6]
    n_params = make_trunk(size, fn).count_params()
    tf.keras.backend.clear_session()
    return f'{n_params // 1000}k-{digest}'


def trunk_path(size, settings, tag, fn):
    """Where the pretrained backbone of these settings is saved (relative to the project root)."""
    return f'models/trunk_{backbone_id(size, settings, fn)}_{tag}.weights.h5'


def check_trunk(path, size, fn):
    """Load `path` into a fresh backbone. Returns the path, or raises an error that says what is wrong."""
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} does not exist (paths are relative to the project folder). '
                                'Pretrain it with MoE_approach.ipynb, or write the path of an existing backbone.')
    try:
        make_trunk(size, fn).load_weights(path)
    except Exception as e:
        raise ValueError(f'{path} does not fit the backbone `backbone` of this notebook '
                         f'({type(e).__name__}: {e}). It was probably saved by a backbone with other layers.') from e
    finally:
        tf.keras.backend.clear_session()
    return path


def make_backbones(size, fn, weights=None, feature_drop=0.3):
    """Backbone functions `fn(inp, augmentation)` for build_cnn / build_moe / build_pmoe / build_mge.
    Every call builds NEW layers, so every model trains its own copy. weights=<file>: the copy starts from
    the pretrained backbone; weights=None: from scratch."""
    def maps(inp, augmentation=None):
        x = augmentation(inp) if augmentation is not None else inp
        trunk = make_trunk(size, fn)
        if weights:
            trunk.load_weights(weights)
        return list(trunk(x))

    def last(inp, augmentation=None):
        return maps(inp, augmentation)[-1]

    def pooled(inp, augmentation=None):
        x = layers.GlobalAveragePooling2D()(last(inp, augmentation))
        return layers.Dropout(feature_drop)(x)

    return Backbones(maps, last, pooled)
