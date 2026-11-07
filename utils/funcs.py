import os
import math
import zipfile

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
                size=2, shuffle=False, seed=0):
    """Plot images from a dataframe in a grid.

    title_cols: a column name or a list of column names to show above each image.
    n: how many images to show (default: all rows).
    shuffle: pick random rows instead of the first ones.
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
        image = Image.open(row[path_col]).convert('RGB')
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
