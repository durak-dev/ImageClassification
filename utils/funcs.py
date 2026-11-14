import os
import math
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SPLITS = ('train', 'val', 'test')
LABELS_FILE = 'labels.csv'                 # the label table inside data/original/labels/
CORRECTED_LABELS_FILE = 'labels_corrected.csv'


def find_label_table(orig_dir, label_file=LABELS_FILE):
    """Path of the label table: <orig_dir>/labels/<label_file> if a labels/ folder
    exists, otherwise <orig_dir>/<label_file> directly."""
    candidates = [os.path.join(orig_dir, 'labels', label_file), os.path.join(orig_dir, label_file)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{label_file} not found. Looked in: " + ", ".join(candidates))


def thumbnail(path, size=16):
    """Tiny color version of an image, rescaled so that its mean is 0 and its standard
    deviation is 1. That removes brightness and contrast; shrinking averages out noise
    and blur. Differences are therefore measured in units of the image's own contrast."""
    image = cv2.imread(path)
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (small - small.mean()) / (small.std() + 1e-6)


def orientations(thumb):
    """The 8 ways to turn an image: 4 rotations, each normal and mirrored.
    Returns an array of shape (8, number of values in the thumbnail)."""
    versions = []
    for k in range(4):
        rotated = np.rot90(thumb, k)
        versions.append(rotated.ravel())
        versions.append(rotated[:, ::-1].ravel())
    return np.array(versions)


def find_duplicate_pairs(df, image_dir, max_diff=0.12, filename_col='filename', chunk=64):
    """Find pairs of images that look the same (also upside down, mirrored, noisier,
    brighter, blurrier).

    Every image becomes a 16x16 color thumbnail (see thumbnail). The 'diff' of two images is
    the root-mean-square difference between their thumbnails, using the rotation/mirror
    that fits best:  diff 0 = identical,  0.12 = on average 12% of the picture's own
    contrast is different,  ~1.4 = unrelated. Pairs with diff <= max_diff are returned
    as a DataFrame with columns a, b (filenames) and diff."""
    thumbs = np.array([thumbnail(os.path.join(image_dir, f)) for f in df[filename_col]])
    n = len(thumbs)
    flat = thumbs.reshape(n, -1) / np.sqrt(thumbs[0].size)           # so that dot product = 1 - diff^2/2

    names = list(df[filename_col])
    rows = []
    for start in range(0, n, chunk):                                 # a chunk of images at a time (memory)
        turned = np.array([orientations(t) for t in thumbs[start:start + chunk]])
        turned = turned.reshape(-1, flat.shape[1]) / np.sqrt(thumbs[0].size)
        similarity = (turned @ flat.T).reshape(-1, 8, n).max(axis=1)  # best of the 8 orientations
        diff = np.sqrt(np.maximum(2 - 2 * similarity, 0))             # RMS difference
        for i, j in zip(*np.where(diff <= max_diff)):
            if j > start + i:                                         # each pair once, not an image with itself
                rows.append((names[start + i], names[j], diff[i, j]))
    return pd.DataFrame(rows, columns=['a', 'b', 'diff'])


def mark_duplicates(df, pairs, filename_col='filename'):
    """Return a copy of df with 'group' (images linked by a pair share an id) and
    'is_duplicate' (True if the image is in a pair). Pass the pairs you accept,
    e.g. pairs[pairs.is_duplicate]."""
    group = {name: i for i, name in enumerate(df[filename_col])}      # every image starts alone
    for a, b in zip(pairs['a'], pairs['b']):
        old, new = group[b], group[a]
        for name in group:
            if group[name] == old:
                group[name] = new                                      # merge b's group into a's

    out = df.copy()
    out['group'] = out[filename_col].map(group)
    out['is_duplicate'] = out.duplicated('group', keep=False)
    return out


def make_splits(orig_dir, manifest_path, duplicates, val_frac=0.15, test_frac=0.15, seed=42,
                label_file=LABELS_FILE, stratify_col='class_id', reuse=True):
    """Decide which split every raw image belongs to, BEFORE any preprocessing.

    Reads the label table (<orig_dir>/labels/<label_file> or <orig_dir>/<label_file>;
    needs a 'filename' column) and the images in <orig_dir>/images/.

    1. `duplicates` is your dataframe after mark_duplicates
       (filename, group, is_duplicate); this function does no image comparison itself.
    2. Every image that has at least one look-alike is QUARANTINED:
       all instances of a duplicate group go to 'train', none to val/test, so
       the same picture can never be in train and in val/test.
    3. The remaining unique images are split in two steps (stratified by
       `stratify_col`): train vs (val+test), then (val+test) -> val / test.
       The val and test sizes are val_frac / test_frac of ALL images, so the
       final ratios hold even though the quarantined images are added to train.

    Returns (and saves to manifest_path) a table: filename, group, is_duplicate, split.
    If manifest_path already exists and reuse=True it is loaded instead, so the
    whole team keeps the same split (delete the file to re-split)."""
    labels = pd.read_csv(find_label_table(orig_dir, label_file))
    if 'filename' not in labels.columns:
        raise ValueError(f"{label_file} needs a 'filename' column")
    if labels['filename'].duplicated().any():
        raise ValueError(f"{label_file} lists some filenames more than once")

    if reuse and os.path.exists(manifest_path):
        manifest = pd.read_csv(manifest_path)
        if set(manifest['filename']) != set(labels['filename']):
            raise ValueError(f"{manifest_path} does not match {label_file} (files were added or "
                             "removed). Delete the manifest to create a new split.")
        print(f"Reusing existing split from {manifest_path}")
        print(manifest['split'].value_counts().to_string())
        return manifest

    image_dir = os.path.join(orig_dir, 'images')
    missing = [f for f in labels['filename'] if not os.path.exists(os.path.join(image_dir, f))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images from {label_file} are missing in "
                                f"{image_dir}, e.g. {missing[:3]}")
    extra = set(os.listdir(image_dir)) - set(labels['filename'])
    if extra:
        print(f"Warning: {len(extra)} files in {image_dir} have no label and are ignored")

    if set(duplicates['filename']) != set(labels['filename']):
        raise ValueError("The duplicate table does not match the label table - rerun mark_duplicates")
    df = labels.merge(duplicates[['filename', 'group', 'is_duplicate']], on='filename', how='left')

    n_total = len(df)
    n_val, n_test = round(val_frac * n_total), round(test_frac * n_total)
    pool = df[~df['is_duplicate']]
    if n_val + n_test >= len(pool):
        raise ValueError("Not enough unique images for the requested val/test sizes")

    def split_off(frame, n, strat_col):
        """Take n rows out of `frame`; returns (rest, taken). Stratified if possible."""
        strat = frame[strat_col] if strat_col in frame.columns else None
        try:
            return train_test_split(frame, test_size=n, random_state=seed, stratify=strat)
        except ValueError as e:                    # e.g. a class with a single image
            print(f"Warning: stratified split not possible ({e}); splitting without stratification")
            return train_test_split(frame, test_size=n, random_state=seed)

    _, temp = split_off(pool, n_val + n_test, stratify_col)     # the rest of the pool is train (default below)
    val_part, test_part = split_off(temp, n_test, stratify_col)

    df['split'] = 'train'                                       # duplicates stay here
    df.loc[val_part.index, 'split'] = 'val'
    df.loc[test_part.index, 'split'] = 'test'

    manifest = df[['filename', 'group', 'is_duplicate', 'split']]
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    groups = df[df['is_duplicate']]
    print(f"{len(groups)} images belong to {groups['group'].nunique()} near-duplicate groups "
          f"-> all quarantined in train")
    if stratify_col in df.columns and len(groups):
        conflicting = (groups.groupby('group')[stratify_col].nunique() > 1).sum()
        if conflicting:
            print(f"Warning: {conflicting} duplicate groups carry DIFFERENT labels - check them")
    print((manifest['split'].value_counts() / n_total).round(3).to_string())
    return manifest


def split_dir(data_dir, split, sub):
    """Path of data/<split>/<sub> (sub = 'images', 'labels' or 'images_corrected')."""
    return os.path.join(data_dir, split, sub)


def materialize_splits(orig_dir, data_dir, manifest, label_file=LABELS_FILE):
    """Create data/{train,val,test}/{images,labels,images_corrected} and copy the
    raw files of every split into images/ and labels/.

    labels/ gets (a) the rows of the label table belonging to that split and
    (b) any per-image label file in <orig_dir>/labels/ (if that folder exists) with the same file
    name stem as the image (e.g. image_1.JPEG <-> image_1.json).
    Files that do not belong in a split (left over from an older split) are
    removed from that split's folders, so a split can never contain another
    split's images. Originals are never modified."""
    labels_path = find_label_table(orig_dir, label_file)
    labels = pd.read_csv(labels_path)
    orig_labels = os.path.dirname(labels_path)
    # per-image label files only exist if there is a labels/ folder next to the table
    side_files = ({os.path.splitext(f)[0]: f for f in os.listdir(orig_labels) if f != label_file}
                  if os.path.basename(orig_labels) == 'labels' else {})

    for split in SPLITS:
        names = list(manifest.loc[manifest['split'] == split, 'filename'])
        keep_images = set(names)
        stems = {os.path.splitext(n)[0] for n in names}
        img_out, lab_out, fix_out = (split_dir(data_dir, split, d)
                                     for d in ('images', 'labels', 'images_corrected'))
        for d in (img_out, lab_out, fix_out):
            os.makedirs(d, exist_ok=True)

        # remove stale files (images moved to another split by a re-split)
        for d, keep in ((img_out, keep_images), (fix_out, keep_images)):
            for f in os.listdir(d):
                if f not in keep:
                    os.remove(os.path.join(d, f))
        for f in os.listdir(lab_out):
            if f not in (label_file, CORRECTED_LABELS_FILE) and os.path.splitext(f)[0] not in stems:
                os.remove(os.path.join(lab_out, f))

        for f in names:                                     # copy raw images
            dst = os.path.join(img_out, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(orig_dir, 'images', f), dst)
            stem = os.path.splitext(f)[0]
            if stem in side_files and not os.path.exists(os.path.join(lab_out, side_files[stem])):
                shutil.copy2(os.path.join(orig_labels, side_files[stem]), lab_out)

        labels[labels['filename'].isin(keep_images)].to_csv(os.path.join(lab_out, label_file), index=False)
        print(f"{split:5s}: {len(names)} images")

    # safety net: no file may exist in two splits
    seen = {}
    for split in SPLITS:
        for f in os.listdir(split_dir(data_dir, split, 'images')):
            if f in seen:
                raise RuntimeError(f"{f} is in both '{seen[f]}' and '{split}'")
            seen[f] = split


def load_splits(data_dir, splits=SPLITS, label_file=LABELS_FILE, image_folder='images'):
    """All partitions in one dataframe: the label tables data/<split>/labels/<label_file>
    plus a 'path' column pointing into data/<split>/<image_folder>/ and a 'split' column.
    Use image_folder='images_corrected' to read the restored images."""
    parts = []
    for split in splits:
        df = pd.read_csv(os.path.join(split_dir(data_dir, split, 'labels'), label_file))
        folder = split_dir(data_dir, split, image_folder)
        df['path'] = df['filename'].map(lambda f: os.path.join(folder, f))
        df['split'] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def first_name(description):
    """'tabby, tabby cat' -> 'tabby'"""
    return str(description).split(',')[0]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_images(data, path_col='path', title_cols=None, n=None, ncols=8,
                size=2, shuffle=False, seed=0, transform=None):
    """Plot images from a dataframe in a grid.

    title_cols: a column name or a list of column names to show above each image.
    n: how many images to show (default: all rows).
    shuffle: pick random rows instead of the first ones.
    transform: optional function applied to each image before showing it,
               e.g. transform=denoise_image to see the corrected version.
    """
    if shuffle:
        data = data.sample(frac=1, random_state=seed)
    if n is not None:
        data = data.head(n)
    if isinstance(title_cols, str):
        title_cols = [title_cols]

    nrows = math.ceil(len(data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.1),
                             squeeze=False)
    axes = axes.flatten()

    for ax in axes:
        ax.axis('off')

    for i in range(len(data)):
        row = data.iloc[i]
        ax = axes[i]
        image = np.array(Image.open(row[path_col]).convert('RGB'))
        if transform is not None:
            image = transform(image)
        ax.imshow(image)
        if title_cols:
            title = '\n'.join(first_name(row[c]) for c in title_cols)
            ax.set_title(title, fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLIP (used to check the labels, not to train)
# ---------------------------------------------------------------------------

def _normalize(output):
    """Make every feature vector length 1 so a dot product = cosine similarity.
    (Newer transformers versions return an object instead of a tensor.)"""
    if not torch.is_tensor(output):
        output = output.pooler_output
    return output / output.norm(dim=-1, keepdim=True)


def clip_text_features(model, processor, device, prompts):
    """One normalised feature vector per text prompt. Shape: (len(prompts), dim)."""
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors='pt', padding=True).to(device)
        features = _normalize(model.get_text_features(**inputs))
    return features.cpu()


def clip_image_features(model, processor, device, paths, batch_size=64):
    """One normalised feature vector per image path. Shape: (len(paths), dim)."""
    all_features = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        images = [Image.open(p).convert('RGB') for p in batch_paths]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors='pt').to(device)
            features = _normalize(model.get_image_features(**inputs))
        all_features.append(features.cpu())
    return torch.cat(all_features)


# ---------------------------------------------------------------------------
# Image quality: detection
# ---------------------------------------------------------------------------
# All metrics are computed on the grayscale image (values 0-255).
# The default thresholds are only starting points - look at the metric
# distributions (e.g. quality.describe()) and the flagged images, then adjust.

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_level(gray, block=8, flat_fraction=0.2):
    """Estimate the noise standard deviation (sigma, in gray levels 0-255).

    Immerkaer's method: convolve with a kernel that cancels smooth image content
    and straight edges, so what remains is mostly noise. Texture (fur, grass,
    foliage) also survives that filter, so we only trust the FLATTEST blocks:
    the image is cut into `block` x `block` tiles and only the `flat_fraction`
    tiles with the weakest response are averaged. In flat areas there is nothing
    but noise, so textured images are no longer mistaken for noisy ones.
    Higher = noisier; roughly 1 for a clean image, 10+ for visibly grainy ones."""
    response = np.abs(cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL,
                                   borderType=cv2.BORDER_REFLECT))[1:-1, 1:-1]
    h = response.shape[0] - response.shape[0] % block
    w = response.shape[1] - response.shape[1] % block
    tiles = response[:h, :w].reshape(h // block, block, w // block, block).mean(axis=(1, 3)).ravel()
    n_flat = max(1, int(len(tiles) * flat_fraction))
    flat = np.sort(tiles)[:n_flat]
    return float(np.sqrt(np.pi / 2) / 6 * flat.mean())


def inspect_images(df, checks=('noise', 'blur', 'contrast'), path_col='path',
                   noise_threshold=2.0, blur_threshold=2.0, contrast_threshold=12.0,
                   high_contrast_threshold=31.0):
    """Run the chosen quality checks on every image in df.

    All scores (and thresholds) are in readable units:
      noise    = grain: noise sigma as % of the 0-255 brightness range  (higher = grainier)
                 ~0.5 clean, 2 slightly grainy, 4+ visibly grainy
      blur     = sharpness: log10 of the Laplacian variance, 1 step = 10x  (LOWER = blurrier)
                 ~1 very blurry, 2 soft, 4 sharp
      contrast = brightness spread (std) as % of the 0-255 range  (lower = flatter)
                 ~10 washed out, 20 normal, 30+ harsh

    checks: any of 'noise', 'blur', 'contrast'.
    noise_threshold: above this -> is_noisy.        blur_threshold: below this -> is_blurry.
    contrast_threshold: below this -> is_low_contrast.
    high_contrast_threshold: above this -> is_high_contrast.
    Returns a DataFrame with the same index as df, holding the path, the scores
    and the boolean flags. Merge it back with: df = df.join(quality.drop(columns='path'))
    """
    rows = []
    for i in range(len(df)):
        path = df[path_col].iloc[i]
        gray = np.array(Image.open(path).convert('L'))   # 'L' = grayscale
        row = {path_col: path}

        if 'noise' in checks:
            row['noise'] = 100 * noise_level(gray) / 255
            row['is_noisy'] = row['noise'] > noise_threshold

        if 'blur' in checks:
            # variance of the Laplacian (edge detector): sharp images have strong edges. LOWER = blurrier
            row['blur'] = float(np.log10(max(cv2.Laplacian(gray, cv2.CV_64F).var(), 1e-6)))
            row['is_blurry'] = row['blur'] < blur_threshold

        if 'contrast' in checks:
            # RMS contrast = std of the pixel values. LOWER = flatter / washed out, HIGHER = harsher
            row['contrast'] = 100 * float(gray.std()) / 255
            row['is_low_contrast'] = row['contrast'] < contrast_threshold
            row['is_high_contrast'] = row['contrast'] > high_contrast_threshold

        rows.append(row)

        if (i + 1) % 10000 == 0:
            print(f"checked {i + 1} / {len(df)} images")

    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Image quality: correction
# ---------------------------------------------------------------------------
# Each function takes an RGB image as a numpy array (height, width, 3) with
# values 0-255 and returns a corrected image of the same shape.
# They can be passed straight to plot_images(..., transform=...).

def denoise_image(image, strength=10, template_window=None, search_window=None):
    """Remove grain with OpenCV's non-local means filter.
    `strength` (h) should be about the noise sigma: higher removes more noise
    but also more fine detail.
    Window sizes default to values scaled to the image size. The OpenCV defaults
    (7 / 21) are tuned for large photos; on a 64x64 image they would average
    over a big part of the whole picture and wash out detail."""
    if template_window is None or search_window is None:
        small = min(image.shape[:2]) <= 128
        template_window = template_window or (3 if small else 7)
        search_window = search_window or (11 if small else 21)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)       # OpenCV expects BGR order
    clean = cv2.fastNlMeansDenoisingColored(bgr, None, strength, strength,
                                            template_window, search_window)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)


def deblur_image(image, amount=1.0, radius=1.0):
    """Sharpen with an 'unsharp mask':
    sharp = image + amount * (image - blurred image).
    The difference (image - blurred) is the edges, so this boosts the edges.
    It cannot recover detail that was never there, it only makes edges crisper."""
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def adjust_contrast(image, clip_limit=2.0, tile_grid=4):
    """Boost local contrast with CLAHE (contrast-limited adaptive histogram
    equalisation) on the lightness channel only, so colours are not shifted.
    The image is split into tile_grid x tile_grid tiles that are equalised
    separately; `clip_limit` caps how much a tile's histogram may be stretched,
    which stops noise in flat areas from being blown up. Higher = stronger."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def reduce_contrast(image, scale=0.8):
    """Tone down harsh contrast: pull the lightness of every pixel towards the
    image's average lightness, L = mean + scale * (L - mean). Only the lightness
    channel changes, so colours are not shifted.
    scale = 1 leaves the image unchanged, 0.8 keeps 80% of the contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0]
    lab[..., 0] = lightness.mean() + scale * (lightness - lightness.mean())
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def severity(ratio, max_ratio):
    """Map a 'how many times worse than the threshold' ratio to a 0-1 severity.

    Uses a log scale: noise and (especially) the Laplacian-variance blur score
    span orders of magnitude, so a linear ratio would put nearly every image at
    either the minimum or the cap. severity = log(ratio) / log(max_ratio):
      ratio = 1 (right at the threshold) -> 0
      ratio = max_ratio (or worse)       -> 1   (clipped, so it never explodes)
    """
    ratio = max(float(ratio), 1.0)
    return float(np.clip(np.log(ratio) / np.log(max_ratio), 0.0, 1.0))


# strength columns written by fix_flagged_images (0 = step not applied)
FIX_COLUMNS = ('denoise_strength', 'denoise_blend', 'contrast_clip', 'contrast_blend',
               'contrast_scale', 'deblur_amount')


def apply_fixes(image, denoise_strength=0, denoise_blend=0, contrast_clip=0, contrast_blend=0,
                contrast_scale=0, deblur_amount=0):
    """Run the corrections with given strengths (0 = skip that step), in the order
    denoise -> contrast -> sharpen. fix_flagged_images works the strengths out per
    image; rebuild_corrected replays the stored ones."""
    if denoise_strength:
        cleaned = denoise_image(image, strength=denoise_strength)
        image = np.clip(denoise_blend * cleaned + (1 - denoise_blend) * image, 0, 255).astype(np.uint8)
    if contrast_clip:
        boosted = adjust_contrast(image, clip_limit=contrast_clip)
        image = np.clip(contrast_blend * boosted + (1 - contrast_blend) * image, 0, 255).astype(np.uint8)
    if contrast_scale:
        image = reduce_contrast(image, scale=contrast_scale)
    if deblur_amount:
        image = deblur_image(image, amount=deblur_amount)
    return image


def fix_flagged_images(df, out_folder='data/images_corrected', path_col='path',
                       noise_threshold=None, blur_threshold=None, contrast_threshold=None,
                       high_contrast_threshold=None,
                       denoise_scale=0.8, denoise_range=(3.0, 15.0), noise_max_ratio=4.0,
                       deblur_range=(0.5, 2.0), blur_max_drop=1.0,
                       clip_range=(1.0, 3.0), contrast_max_ratio=2.0, min_contrast_scale=0.6,
                       jpeg_quality=95, copy_unflagged=True):
    """Apply corrections to every flagged image and save the result as a new file.

    Uses the flag columns from inspect_images (is_noisy, is_blurry, is_low_contrast,
    is_high_contrast) and the score columns 'noise' / 'blur' / 'contrast', in the
    readable units of inspect_images. Use the SAME thresholds you gave to inspect_images.

    Every step is OFF unless you pass its threshold:
      noise_threshold -> denoise,            blur_threshold -> sharpen,
      contrast_threshold -> boost contrast,  high_contrast_threshold -> reduce contrast.
    e.g. fix_flagged_images(df, out, noise_threshold=2.0) only denoises.
    An enabled step runs only for images whose flag is set, so clearing a flag in the
    notebook (e.g. no contrast fix for noisy images) switches it off for that image.

    Denoising (noise = grain %, converted back to the sigma in gray levels):
      * filter strength h = denoise_scale * sigma, clipped to denoise_range (gray levels).
        Non-local means expects h close to the noise sigma, so the strength
        follows the measured noise directly.
      * the denoised image is then BLENDED with the original:
        out = alpha * denoised + (1 - alpha) * original, where alpha is the
        severity (0-1, log scale, reaching 1 at noise_max_ratio x threshold),
        floored at 0.3. Mild cases keep most of their detail and a wrong noise
        estimate cannot destroy an image.
    Sharpening (blur = sharpness log10, lower = blurrier):
      * amount = lo + severity * (hi - lo), severity = (threshold - sharpness) / blur_max_drop,
        clipped to 0-1: full strength once an image is blur_max_drop log steps (default 1 = 10x)
        below the threshold. Reduced by 25% when the image is also noisy, because noise
        inflates the blur score.
    Contrast (contrast = brightness spread %, lower = flatter):
      * CLAHE on the lightness channel; its clip limit goes from clip_range[0]
        to clip_range[1] with a log-scale severity of threshold / contrast, and
        the result is blended with the original by the same severity (floored
        at 0.3) so a barely-low-contrast image is only nudged.
    Too much contrast (contrast > high_contrast_threshold):
      * reduce_contrast with scale = high_contrast_threshold / contrast, i.e. the
        image's contrast is brought back down to about the threshold; the scale is
        never below min_contrast_scale, so no image loses more than 40% by default.
    Order: denoise -> contrast -> sharpen, so later steps do not amplify grain.

    With copy_unflagged=True the unflagged images are copied unchanged into
    out_folder, so out_folder holds the complete dataset (e.g. data/train/images_corrected).
    The original files are not touched. Returns a copy of df where 'path' points
    to the file in out_folder, plus 'was_fixed', 'orig_path',
    'denoise_strength', 'denoise_blend', 'contrast_clip', 'contrast_scale' and
    'deblur_amount' (0 = step not applied).
    """
    os.makedirs(out_folder, exist_ok=True)
    df = df.copy()
    df['orig_path'] = df[path_col]
    new_paths = df[path_col].tolist()
    was_fixed = np.zeros(len(df), dtype=bool)
    used = {c: np.zeros(len(df)) for c in FIX_COLUMNS}

    if all(t is None for t in (noise_threshold, blur_threshold, contrast_threshold, high_contrast_threshold)):
        print("Warning: no threshold given, so no step is enabled - images are only copied")

    def flag(row, name, threshold, score):
        """True if this step is enabled (threshold given) and the image is flagged for it."""
        return threshold is not None and name in row.index and score in row.index and bool(row[name])

    for i in range(len(df)):
        row = df.iloc[i]
        noisy = flag(row, 'is_noisy', noise_threshold, 'noise')
        blurry = flag(row, 'is_blurry', blur_threshold, 'blur')
        flat = flag(row, 'is_low_contrast', contrast_threshold, 'contrast')
        harsh = flag(row, 'is_high_contrast', high_contrast_threshold, 'contrast')

        if not (flat or harsh or noisy or blurry):
            if copy_unflagged:                          # nothing to fix: copy unchanged
                dst = os.path.join(out_folder, os.path.basename(row[path_col]))
                shutil.copy2(row[path_col], dst)
                new_paths[i] = dst
            continue

        s = dict.fromkeys(FIX_COLUMNS, 0.0)             # strengths for this image

        if noisy:
            sigma = float(row['noise']) * 255 / 100                 # grain % -> gray levels for the filter
            s['denoise_strength'] = float(np.clip(denoise_scale * sigma, *denoise_range))
            s['denoise_blend'] = max(0.3, severity(float(row['noise']) / noise_threshold, noise_max_ratio))

        if flat:
            sev = severity(contrast_threshold / max(float(row['contrast']), 1e-6), contrast_max_ratio)
            s['contrast_clip'] = clip_range[0] + sev * (clip_range[1] - clip_range[0])
            s['contrast_blend'] = max(0.3, sev)

        if harsh:
            s['contrast_scale'] = float(np.clip(high_contrast_threshold / float(row['contrast']),
                                                min_contrast_scale, 1.0))

        if blurry:
            sev = float(np.clip((blur_threshold - float(row['blur'])) / blur_max_drop, 0.0, 1.0))
            amount = deblur_range[0] + sev * (deblur_range[1] - deblur_range[0])
            if 'is_noisy' in row.index and bool(row['is_noisy']):   # noise inflates the blur score
                amount *= 0.75
            s['deblur_amount'] = amount

        image = apply_fixes(np.array(Image.open(row[path_col]).convert('RGB')), **s)
        new_path = os.path.join(out_folder, os.path.basename(row[path_col]))
        Image.fromarray(image).save(new_path, quality=jpeg_quality)
        new_paths[i] = new_path
        was_fixed[i] = True
        for c in FIX_COLUMNS:
            used[c][i] = s[c]

    df[path_col] = new_paths
    df['was_fixed'] = was_fixed
    for c in FIX_COLUMNS:
        df[c] = used[c]

    print(f"fixed {was_fixed.sum()} of {len(df)} images -> saved in '{out_folder}'")
    return df


def rebuild_corrected(fixes, data_dir, labels_corrected=None, jpeg_quality=95, overwrite=False):
    """Recreate data/{split}/images_corrected/ (and labels_corrected.csv) on a fresh
    machine such as Colab, without re-running the quality check or CLIP.

    fixes: the df_fixed checkpoint (filename, split, was_fixed + the FIX_COLUMNS strengths).
           Every image gets exactly the corrections stored for it; the rest are copied.
    labels_corrected: optional table (filename, split, class_id, description), written as
           data/{split}/labels/labels_corrected.csv. It is written last, so its presence
           means the rebuild finished.
    Needs data/{split}/images/ first: run materialize_splits(orig_dir, data_dir, manifest).
    Files that already exist are skipped (overwrite=False), so an interrupted run can resume."""
    fixes = fixes.copy()
    if 'contrast_blend' not in fixes.columns:           # older checkpoints: same formula as fix_flagged_images
        fixes['contrast_blend'] = np.where(fixes['contrast_clip'] > 0,
                                           np.maximum(0.3, (fixes['contrast_clip'] - 1.0) / 2.0), 0.0)

    for split, part in fixes.groupby('split'):
        src_dir, out_dir = split_dir(data_dir, split, 'images'), split_dir(data_dir, split, 'images_corrected')
        missing = set(part['filename']) - set(os.listdir(src_dir))
        if missing:
            raise RuntimeError(f"{len(missing)} images of the checkpoint are not in {src_dir} "
                               "(was the checkpoint made with a different split_manifest.csv?)")
        os.makedirs(out_dir, exist_ok=True)
        written = 0
        for _, row in part.iterrows():
            src, dst = os.path.join(src_dir, row['filename']), os.path.join(out_dir, row['filename'])
            if os.path.exists(dst) and not overwrite:
                continue
            if bool(row['was_fixed']):
                image = np.array(Image.open(src).convert('RGB'))
                image = apply_fixes(image, **{c: float(row.get(c, 0.0)) for c in FIX_COLUMNS})
                Image.fromarray(image).save(dst, quality=jpeg_quality)
            else:
                shutil.copy2(src, dst)
            written += 1
        print(f"{split:5s}: {written} images written, {int(part['was_fixed'].sum())} of {len(part)} corrected -> {out_dir}")

    if labels_corrected is not None:
        for split, part in labels_corrected.groupby('split'):
            part.drop(columns='split').to_csv(
                os.path.join(split_dir(data_dir, split, 'labels'), CORRECTED_LABELS_FILE), index=False)
        print('labels_corrected.csv written for', ', '.join(sorted(labels_corrected['split'].unique())))


def plot_history(history):
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set(title="Loss", xlabel="epoch"); ax[0].legend()
    if "val_f1" in h:
        ax[1].plot(h["val_f1"], color="C2", marker="o")
        best = int(np.argmax(h["val_f1"]))
        ax[1].set(title=f"Val macro-F1 (best {h['val_f1'][best]:.3f} @ epoch {best})", xlabel="epoch")
    plt.tight_layout(); plt.show()