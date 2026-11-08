import os
import math

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_device():
    """'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset(data_dir):
    """Read labels.csv from the repository's data folder and return it as a
    dataframe with an extra 'path' column pointing at each image file.

    Expected layout:  <data_dir>/labels.csv  and  <data_dir>/images/<filename>
    """
    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    df['path'] = df['filename'].map(lambda f: os.path.join(data_dir, 'images', f))
    return df


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


def plot_class_counts(df, label_col='class_id', desc_col='description', n_samples=4):
    """Bar chart of how many images each class has, sorted largest first."""
    counts = df.groupby(label_col).size().sort_values(ascending=False)
    desc = df.drop_duplicates(label_col).set_index(label_col)[desc_col]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(counts)), counts.values, width=1.0)

    
    # label a few bars along the x axis so the plot is readable
    positions = [int((i+0.5)*2) * len(counts)//(n_samples * 2) for i in range(n_samples)]
    labels = [first_name(desc[counts.index[i]]) for i in positions]
    plt.xticks(positions, labels, rotation=45, ha='right')

    plt.xlabel(f'{len(counts)} classes (sorted by size)')
    plt.ylabel('Images')
    plt.title('Images per class')
    plt.tight_layout()
    plt.show()
    return counts


def plot_confusion(y_true, y_pred, n_classes):
    """Plot a row-normalised confusion matrix and return it.
    Row i shows where the images of class i ended up (fractions sum to 1)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    cmn = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(cmn, cmap='viridis')
    plt.colorbar(label='Fraction of class')
    plt.xlabel('CLIP label')
    plt.ylabel('Original label')
    plt.title(f'Overall agreement: {np.mean(y_true == y_pred):.2%}')
    plt.show()
    return cmn


# ---------------------------------------------------------------------------
# CLIP (used to check the labels, not to train)
# ---------------------------------------------------------------------------

def load_clip(name='openai/clip-vit-base-patch32'):
    """Load a CLIP model and its processor. Returns (model, processor, device)."""
    device = get_device()
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor, device


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


def class_neighbor_ranks(text_features):
    """For every pair of classes (i, j), how close j is to i in meaning.
    ranks[i, j] == 1 means j is the class most similar to i, 2 the second most, etc.
    Used to tell 'CLIP confused two similar classes' apart from 'label looks wrong'."""
    T = text_features.float().numpy()
    T = T - T.mean(axis=0)                      # remove the shared "a photo of a" part
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    similarity = T @ T.T
    np.fill_diagonal(similarity, -np.inf)       # a class is not its own neighbour
    ranks = (-similarity).argsort(axis=1).argsort(axis=1) + 1
    return ranks


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


def blur_level(gray):
    """How sharp the image is: the variance of the Laplacian (an edge detector).
    Sharp images have many strong edges -> high value. LOWER = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def contrast_level(gray):
    """How much the image uses the available tonal range: the standard deviation
    of the pixel values (RMS contrast). Low = flat / washed-out / foggy.
    LOWER = less contrast."""
    return float(gray.std())


def inspect_images(df, checks=('noise', 'blur', 'contrast'), path_col='path',
                   noise_threshold=5, blur_threshold=100, contrast_threshold=30):
    """Run the chosen quality checks on every image in df.

    checks: any of 'noise', 'blur', 'contrast'.
    Returns a DataFrame with the same index as df, holding the path, the metric
    values and the boolean flags (is_noisy, is_blurry, is_low_contrast).
    Merge it back with: df = df.join(quality.drop(columns='path'))
    """
    rows = []
    for i in range(len(df)):
        path = df[path_col].iloc[i]
        gray = np.array(Image.open(path).convert('L'))   # 'L' = grayscale
        row = {path_col: path}

        if 'noise' in checks:
            row['noise'] = noise_level(gray)
            row['is_noisy'] = row['noise'] > noise_threshold

        if 'blur' in checks:
            row['blur'] = blur_level(gray)
            row['is_blurry'] = row['blur'] < blur_threshold

        if 'contrast' in checks:
            row['contrast'] = contrast_level(gray)
            row['is_low_contrast'] = row['contrast'] < contrast_threshold

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


def fix_flagged_images(df, out_folder='dataset_fixed', path_col='path',
                       noise_threshold=5.0, blur_threshold=100.0, contrast_threshold=30.0,
                       denoise_scale=0.8, denoise_range=(3.0, 15.0), noise_max_ratio=4.0,
                       deblur_range=(0.5, 2.0), blur_max_ratio=10.0,
                       clip_range=(1.0, 3.0), contrast_max_ratio=2.0,
                       jpeg_quality=95):
    """Apply corrections to every flagged image and save the result as a new file.

    Uses the flag columns from inspect_images (is_noisy, is_blurry,
    is_low_contrast) and the metric columns 'noise' / 'blur' / 'contrast'. Use the SAME thresholds
    you gave to inspect_images.

    Denoising (noise = estimated sigma, see noise_level):
      * filter strength h = denoise_scale * sigma, clipped to denoise_range.
        Non-local means expects h close to the noise sigma, so the strength
        follows the measured noise directly.
      * the denoised image is then BLENDED with the original:
        out = alpha * denoised + (1 - alpha) * original, where alpha is the
        severity (0-1, log scale, reaching 1 at noise_max_ratio x threshold),
        floored at 0.3. Mild cases keep most of their detail and a wrong noise
        estimate cannot destroy an image.
    Sharpening (blur = Laplacian variance, lower = blurrier):
      * amount = lo + severity * (hi - lo) with severity from threshold / blur on
        a log scale (the score spans orders of magnitude); reduced by 25% when
        the image is also noisy, because noise inflates the blur score.
    Contrast (contrast = std of the pixel values, lower = flatter):
      * CLAHE on the lightness channel; its clip limit goes from clip_range[0]
        to clip_range[1] with a log-scale severity of threshold / contrast, and
        the result is blended with the original by the same severity (floored
        at 0.3) so a barely-low-contrast image is only nudged.
    Order: denoise -> contrast -> sharpen, so later steps do not amplify grain.

    The original files are not touched. Returns a copy of df where 'path' points
    to the fixed file for corrected images, plus 'was_fixed', 'orig_path',
    'denoise_strength', 'denoise_blend', 'contrast_clip' and 'deblur_amount'
    (0 = step not applied).
    """
    os.makedirs(out_folder, exist_ok=True)
    df = df.copy()
    df['orig_path'] = df[path_col]
    new_paths = df[path_col].tolist()
    was_fixed = np.zeros(len(df), dtype=bool)
    used_denoise = np.zeros(len(df))
    used_blend = np.zeros(len(df))
    used_deblur = np.zeros(len(df))
    used_clip = np.zeros(len(df))

    def flag(row, name):
        return name in row.index and bool(row[name])

    for i in range(len(df)):
        row = df.iloc[i]
        flat = flag(row, 'is_low_contrast') and 'contrast' in row.index
        noisy = flag(row, 'is_noisy') and 'noise' in row.index
        blurry = flag(row, 'is_blurry') and 'blur' in row.index

        if not (flat or noisy or blurry):
            continue                                    # nothing to fix, keep original

        image = np.array(Image.open(row[path_col]).convert('RGB'))

        if noisy:
            sigma = float(row['noise'])
            h = float(np.clip(denoise_scale * sigma, *denoise_range))
            alpha = max(0.3, severity(sigma / noise_threshold, noise_max_ratio))
            cleaned = denoise_image(image, strength=h)
            image = np.clip(alpha * cleaned + (1 - alpha) * image, 0, 255).astype(np.uint8)
            used_denoise[i], used_blend[i] = h, alpha

        if flat:
            sev = severity(contrast_threshold / max(float(row['contrast']), 1e-6), contrast_max_ratio)
            clip = clip_range[0] + sev * (clip_range[1] - clip_range[0])
            alpha = max(0.3, sev)
            boosted = adjust_contrast(image, clip_limit=clip)
            image = np.clip(alpha * boosted + (1 - alpha) * image, 0, 255).astype(np.uint8)
            used_clip[i] = clip

        if blurry:
            sev = severity(blur_threshold / max(float(row['blur']), 1e-6), blur_max_ratio)
            amount = deblur_range[0] + sev * (deblur_range[1] - deblur_range[0])
            if noisy:
                amount *= 0.75
            used_deblur[i] = amount
            image = deblur_image(image, amount=amount)

        new_path = os.path.join(out_folder, os.path.basename(row[path_col]))
        Image.fromarray(image).save(new_path, quality=jpeg_quality)
        new_paths[i] = new_path
        was_fixed[i] = True

    df[path_col] = new_paths
    df['was_fixed'] = was_fixed
    df['denoise_strength'] = used_denoise
    df['denoise_blend'] = used_blend
    df['deblur_amount'] = used_deblur
    df['contrast_clip'] = used_clip

    print(f"fixed {was_fixed.sum()} of {len(df)} images -> saved in '{out_folder}'")
    return df
