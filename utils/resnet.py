"""Helpers around the backbone function `backbone` that is written in the notebooks.

The backbone itself lives in the notebook, so it is easy to change on the fly. What has to behave the same in
both notebooks sits here: saving / loading / checking a pretrained backbone and building the copies the models
train. Every function takes the backbone function `fn(inp) -> [s1, s2, s3]` as an argument.
"""
import hashlib
import inspect
import json
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers

from utils.MoE import res_block


def make_trunk(size, fn):
    """The backbone function as one Keras model: image (already augmented) -> [s1, s2, s3]."""
    inp = tf.keras.Input((size, size, 3))
    maps = fn(inp)
    assert isinstance(maps, (list, tuple)) and len(maps) == 3, "backbone must return a list of 3 feature maps [s1, s2, s3]"
    return tf.keras.Model(inp, list(maps), name='trunk')


# ---------------------------------------------------------------- saved backbones
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


def trunk_name(path):
    """'models/trunk_6072k-abc123_aug.weights.h5' -> 'trunk_6072k-abc123_aug' (used in log names)."""
    return os.path.basename(path).replace('.weights.h5', '')


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


# ---------------------------------------------------------------- backbone functions for the model builders
Backbones = namedtuple('Backbones', 'maps last pooled')
Backbones.__doc__ = """maps:   [s1, s2, s3] (MGE)
last:   s3, the 8x8x256 map before pooling (pMoE)
pooled: s3 pooled to a 256-d vector + feature dropout (CNN, MoE)"""


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
