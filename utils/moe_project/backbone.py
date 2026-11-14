"""The shared conv feature extractor.

res_block() and backbone() are copied from MoE.py and behave identically.
The two new functions below build the SAME network but stop earlier:
  backbone_feature_map()  -> the (H, W, 256) map BEFORE global pooling (for pMoE)
  backbone_multi_depth()  -> three maps, from the 64, 128 and 256 channel stages (for MGE)
They are written out in full (not shared through a helper) so you can read each one top to bottom.
"""
from tensorflow.keras import layers


def res_block(x, filters, downsample=True):
    stride = 2 if downsample else 1
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))


def backbone(inp, augmentation=None, drop=0.1):
    """Shared conv feature extractor -> 256 features. Used by ALL models so the comparison is fair."""
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    x = res_block(x, 64)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 128)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 256)
    x = layers.GlobalAveragePooling2D()(x)
    return layers.Dropout(drop * 2.5)(x)                             # feature dropout


def backbone_feature_map(inp, augmentation=None, drop=0.1):
    """Same network as backbone(), but returns the feature map BEFORE GlobalAveragePooling2D.
    Shape: (batch, H, W, 256). Each of the H*W positions is one "patch" for pMoE.
    The feature dropout of backbone() is left out here: it acts on pooled vectors,
    and there is no pooled vector in this version."""
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    x = res_block(x, 64)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 128)
    x = layers.SpatialDropout2D(drop)(x)
    x = res_block(x, 256)
    return x                                                         # (batch, H, W, 256)


def backbone_multi_depth(inp, augmentation=None, drop=0.1):
    """Same network again, but returns THREE feature maps (a list), one per stage:
         [ (H/2, W/2, 64),  (H/4, W/4, 128),  (H/8, W/8, 256) ]
    Each one is the output of a res_block, taken before the spatial dropout that follows it.
    The experts of the MGE model attach to these (each pools its own map)."""
    x = augmentation(inp) if augmentation is not None else inp
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)     # stem
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = res_block(x, 32, downsample=False)
    stage_64 = res_block(x, 64)
    x = layers.SpatialDropout2D(drop)(stage_64)
    stage_128 = res_block(x, 128)
    x = layers.SpatialDropout2D(drop)(stage_128)
    stage_256 = res_block(x, 256)
    return [stage_64, stage_128, stage_256]


def n_patches(size):
    """How many spatial positions the final feature map has for a size x size image.
    The backbone halves the size 3 times (rounding up, because padding='same'),
    so size 64 -> 8 x 8 = 64 patches, size 32 -> 4 x 4 = 16 patches."""
    side = size
    for _ in range(3):
        side = -(-side // 2)          # ceil(side / 2)
    return side * side
