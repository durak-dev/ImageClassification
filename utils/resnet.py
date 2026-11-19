"""Helpers for ONE saved, frozen backbone that every head is trained on (MoE_comparisons.ipynb, pMoE_MGE_finetune_v2.ipynb).

The backbone function `fn(inp) -> [s1, s2, s3]` is written in the notebook. A trained backbone ("trunk") is saved as a
complete Keras model (`<name>.keras`: layers AND weights), so a notebook that loads it does not need the code that built
it. You choose a backbone by writing its path.
"""
import os
import time
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers


def norm_path(path):
    """Accept a Windows-style path (backslashes) on Linux/Colab, and a path inside the project folder.
    The result uses '/' and is relative to the project folder, which is also how the file is named on GitHub."""
    p = str(path).replace(chr(92), '/')
    root = os.getcwd().replace(chr(92), '/').rstrip('/') + '/'
    return p[len(root):] if p.startswith(root) else p


def make_trunk(size, fn):
    """The backbone function as one Keras model: image (already augmented) -> [s1, s2, s3]."""
    inp = tf.keras.Input((size, size, 3))
    maps = fn(inp)
    assert isinstance(maps, (list, tuple)) and len(maps) == 3, "backbone must return a list of 3 feature maps [s1, s2, s3]"
    return tf.keras.Model(inp, list(maps), name='trunk')


def new_trunk_path(models_dir, trunk):
    """Automatic, unique file name for a new backbone: <models_dir>/backbone_<params>k_<date-time>[_fp16].keras
    (_fp16 when the notebook runs with mixed_float16: the layers are then saved in fp16 and load that way)."""
    tag = '_fp16' if tf.keras.mixed_precision.global_policy().name == 'mixed_float16' else ''
    return f'{models_dir}/backbone_{trunk.count_params() // 1000}k_{time.strftime("%Y%m%d-%H%M%S")}{tag}.keras'


def trunk_name(path):
    """'dir/backbone_6072k_20251118-210000.keras' -> 'backbone_6072k_20251118-210000' (used in log names)."""
    name = norm_path(path).rsplit('/', 1)[-1]
    for ext in ('.weights.h5', '.keras'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def save_trunk(trunk, path):
    """Save layers + weights. Refuses to overwrite: a backbone other runs depend on must not change silently."""
    if os.path.exists(path):
        raise FileExistsError(f'{path} already exists. Choose another name, or delete the file on purpose.')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    trunk.save(path)
    return path


def load_trunk(path):
    """Load a saved backbone and freeze it. Raises an error that says what is wrong."""
    path = norm_path(path)
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


Backbones = namedtuple('Backbones', 'maps last pooled')
Backbones.__doc__ = """maps:   [s1, s2, s3] (MGE)
last:   s3, the deepest map before pooling (pMoE)
pooled: s3 pooled to one vector + feature dropout (CNN, MoE)"""


def make_frozen_backbones(path, feature_drop=0.3, size=None):
    """Backbone functions `fn(inp, augmentation)` for build_cnn / build_moe / build_pmoe / build_mge.
    The backbone is FROZEN and run in inference mode (batch norm uses its saved statistics, no dropout inside it),
    so every head is trained on exactly the same features and the file is never changed.
    size: if given, a backbone trained on another image size is rejected with a clear error."""
    def maps(inp, augmentation=None):
        x = augmentation(inp) if augmentation is not None else inp
        trunk = load_trunk(path)
        if size is not None and trunk.input_shape[1] != size:
            raise ValueError(f'{path} was trained on {trunk.input_shape[1]}x{trunk.input_shape[1]} images, but SIZE = {size}')
        return list(trunk(x, training=False))

    def last(inp, augmentation=None):
        return maps(inp, augmentation)[-1]

    def pooled(inp, augmentation=None):
        x = layers.GlobalAveragePooling2D()(last(inp, augmentation))
        return layers.Dropout(feature_drop)(x)

    return Backbones(maps, last, pooled)
