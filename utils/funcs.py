import math
from PIL import Image
import matplotlib.pyplot as plt

def plot_images(data, path_col='path', title_cols=None, n=None, ncols=8,
                size=2, shuffle=False, seed=0, sep='\n'):
    """Plot images from a dataframe.
    title_cols: a column name or list of names; joined with `sep`.
    n: number of images (default all); shuffle: random sample."""
    if shuffle:
        data = data.sample(frac=1, random_state=seed)
    if n:
        data = data.head(n)
    if isinstance(title_cols, str):
        title_cols = [title_cols]

    nrows = math.ceil(len(data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.1),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, r) in zip(axes.flat, data.iterrows()):
        with Image.open(r[path_col]) as im:
            ax.imshow(im.convert('RGB'))
        if title_cols:
            ax.set_title(sep.join(str(r[c]).split(',')[0] for c in title_cols), fontsize=8)
    plt.tight_layout(); plt.show()
