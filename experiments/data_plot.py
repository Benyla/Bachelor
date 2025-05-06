import os
import random

import pandas as pd
import matplotlib.pyplot as plt
import torch

# ─── CONFIG ──────────────────────────────────────────────────────────────────

# Paths
metadata_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv"
image_dir     = "/work3/s224194/cell_images_processed"
base_dir      = os.getcwd()
plot_dir      = os.path.join(base_dir, "plots")

# Ensure output directory exists
os.makedirs(plot_dir, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

# Metadata
df = pd.read_csv(metadata_path)
moa_counts = df["moa"].value_counts()

# Random image
pt_files = [f for f in os.listdir(image_dir) if f.endswith(".pt")]
selected_file = random.choice(pt_files)
img_t = torch.load(os.path.join(image_dir, selected_file))
img_np = img_t.numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
img_rgb = img_np.transpose(1, 2, 0)

# ─── PLOTTING FUNCTION ────────────────────────────────────────────────────────

def make_and_save(fig_name, plot_fn):
    """
    Create a figure, call `plot_fn(ax_dist, ax_img)` to draw,
    then save to plots/{fig_name}.png.
    """
    fig, (ax_dist, ax_img) = plt.subplots(1, 2, figsize=(12, 5))
    plot_fn(ax_dist, ax_img)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, fig_name + ".png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")

# ─── 1) Horizontal bar chart + image ──────────────────────────────────────────

def plot_barh(ax_dist, ax_img):
    counts = moa_counts.sort_values(ascending=True)
    counts.plot(kind="barh", ax=ax_dist)
    ax_dist.set_title("MOA Distribution (barh)")
    ax_dist.set_xlabel("Number of Instances")
    ax_dist.set_ylabel("MOA Class")
    ax_dist.grid(True)
    # image
    ax_img.imshow(img_rgb)
    ax_img.set_title(f"Random Image:\n{selected_file}")
    ax_img.axis("off")

make_and_save("moa_barh_with_image", plot_barh)

# ─── 2) Pie chart + image ─────────────────────────────────────────────────────

def plot_pie(ax_dist, ax_img):
    moa_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        labeldistance=1.1,
        ax=ax_dist
    )
    ax_dist.set_ylabel("")
    ax_dist.set_title("MOA Composition (pie)")
    # image
    ax_img.imshow(img_rgb)
    ax_img.set_title(f"Random Image:\n{selected_file}")
    ax_img.axis("off")

make_and_save("moa_pie_with_image", plot_pie)

# ─── 3) Donut chart + image ───────────────────────────────────────────────────

def plot_donut(ax_dist, ax_img):
    moa_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.4),
        textprops=dict(color="white"),
        ax=ax_dist
    )
    ax_dist.set_ylabel("")
    ax_dist.set_title("MOA Composition (donut)")
    # image
    ax_img.imshow(img_rgb)
    ax_img.set_title(f"Randomly Sampled Cell Image")
    ax_img.axis("off")

make_and_save("moa_donut_with_image", plot_donut)

