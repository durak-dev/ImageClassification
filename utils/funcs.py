import os
import math
import zipfile

import cv2
import gdown
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

def download_dataset(url, zip_name='dataset.zip', folder='dataset_extracted'):
    """Download the dataset zip from a Google Drive share link, extract it,
    and return labels.csv as a dataframe with an extra 'path' column."""
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'

    if not os.path.exists(zip_name):
        print("Downloading dataset...")
        gdown.download(download_url, zip_name, quiet=True)

    if not os.path.exists(folder):
        print("Extracting files...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(folder)

    print("Extracted contents:", os.listdir(folder))

    df = pd.read_csv(os.path.join(folder, 'labels.csv'))
    df['path'] = folder + '/images/' + df['filename']
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


def plot_class_counts(df, label_col='class_id', desc_col='description'):
    """Bar chart of how many images each class has, sorted largest first."""
    counts = df.groupby(label_col).size().sort_values(ascending=False)
    desc = df.drop_duplicates(label_col).set_index(label_col)[desc_col]

    plt.figure(figsize=(14, 5))
    plt.bar(range(len(counts)), counts.values, width=1.0)

    # label a few bars along the x axis so the plot is readable
    positions = [len(counts) // 8, 3 * len(counts) // 8, 5 * len(counts) // 8, 7 * len(counts) // 8]
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def noise_level(gray):
    """How grainy the image is: the typical difference between the image and
    a median-blurred copy of itself. Median blur removes speckle noise but keeps
    edges, so what is left over is mostly noise. Higher = noisier.

    We take the MEDIAN of the differences (not the average), so a few strong
    edges or thin lines don't count as noise - noise is everywhere, edges are not.
    The 1.4826 just turns the result into a standard-deviation-like number."""
    smooth = cv2.medianBlur(gray, 3)
    difference = np.abs(gray.astype(np.float32) - smooth.astype(np.float32))
    return float(1.4826 * np.median(difference))


def blur_level(gray):
    """How sharp the image is: the variance of the Laplacian (an edge detector).
    Sharp images have many strong edges -> high value. LOWER = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_level(gray):
    """Average pixel value: 0 = completely black, 255 = completely white."""
    return float(gray.mean())


def inspect_images(df, checks=('noise', 'blur', 'brightness'), path_col='path',
                   noise_threshold=10, blur_threshold=100,
                   dark_threshold=50, bright_threshold=205):
    """Run the chosen quality checks on every image in df.

    checks: any of 'noise', 'blur', 'brightness'.
    Returns a DataFrame with the same index as df, holding the path, the metric
    values and the boolean flags (is_noisy, is_blurry, is_dark, is_bright).
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

        if 'brightness' in checks:
            row['brightness'] = brightness_level(gray)
            row['is_dark'] = row['brightness'] < dark_threshold
            row['is_bright'] = row['brightness'] > bright_threshold

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

def denoise_image(image, strength=10):
    """Remove grain with OpenCV's non-local means filter.
    Higher strength removes more noise but also more fine detail."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)       # OpenCV expects BGR order
    clean = cv2.fastNlMeansDenoisingColored(bgr, None, strength, strength, 7, 21)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)


def deblur_image(image, amount=1.0, radius=1.0):
    """Sharpen with an 'unsharp mask':
    sharp = image + amount * (image - blurred image).
    The difference (image - blurred) is the edges, so this boosts the edges.
    It cannot recover detail that was never there, it only makes edges crisper."""
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def adjust_brightness(image, target=128):
    """Move the average brightness towards `target` with gamma correction.
    Gamma brightens/darkens the mid-tones while keeping black black and
    white white, so it looks more natural than just adding a constant."""
    mean = np.array(Image.fromarray(image).convert('L')).mean()
    mean = min(max(mean, 1), 254)                        # avoid log(0)
    gamma = np.log(target / 255) / np.log(mean / 255)   # solves (mean/255)**gamma = target/255
    corrected = 255 * (image / 255) ** gamma
    return corrected.astype(np.uint8)


def fix_flagged_images(df, out_folder='dataset_fixed', path_col='path',
                       denoise_strength=10, deblur_amount=1.0, brightness_target=128):
    """Apply the corrections to every flagged image and save the result as a new file.

    Uses the flag columns from inspect_images (is_noisy, is_blurry, is_dark,
    is_bright); a missing flag column is simply skipped.
    The fixes run in this order: brightness -> denoise -> deblur.
    (Denoise before sharpening, otherwise sharpening makes the noise stronger.)

    The original files are not touched. Returns a copy of df where 'path' points
    to the fixed file for corrected images, plus a 'was_fixed' column.
    """
    os.makedirs(out_folder, exist_ok=True)
    df = df.copy()
    df['was_fixed'] = False

    def flag(row, name):
        return name in row and bool(row[name])

    for i in range(len(df)):
        row = df.iloc[i]
        too_dark_or_bright = flag(row, 'is_dark') or flag(row, 'is_bright')
        noisy = flag(row, 'is_noisy')
        blurry = flag(row, 'is_blurry')

        if not (too_dark_or_bright or noisy or blurry):
            continue                                    # nothing to fix, keep original

        image = np.array(Image.open(row[path_col]).convert('RGB'))
        if too_dark_or_bright:
            image = adjust_brightness(image, target=brightness_target)
        if noisy:
            image = denoise_image(image, strength=denoise_strength)
        if blurry:
            image = deblur_image(image, amount=deblur_amount)

        new_path = os.path.join(out_folder, os.path.basename(row[path_col]))
        Image.fromarray(image).save(new_path, quality=95)

        df.iloc[i, df.columns.get_loc(path_col)] = new_path
        df.iloc[i, df.columns.get_loc('was_fixed')] = True

    print(f"fixed {df['was_fixed'].sum()} of {len(df)} images -> saved in '{out_folder}'")
    return df
