# visualize_features.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

# -------------------------
# User settings
# -------------------------
CSV_PATH = "/Users/xyx/Documents/EE301 Signal&Systems Project/ESC_22/features_robust.csv"   # path to your CSV
OUT_DIR = "/Users/xyx/Documents/EE301 Signal&Systems Project/ESC_22/feature_plots"         # where plots will be saved
TOP_N_CLASSES = 5                 # how many classes to show in multi-class plots
SAMPLE_FOR_PCA = 1000             # if dataset large, subsample for PCA scatter
RANDOM_STATE = 123
# -------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Load
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples, {df.shape[1]} columns")

# Ensure label exists
if "label" not in df.columns:
    raise RuntimeError("CSV must contain a 'label' column")

labels = df['label'].astype(str)

# helper to safe-get columns
def get_col(name):
    return name if name in df.columns else None

# ---------- Identify typical features ----------
time_features = [c for c in df.columns if c.startswith(("zcr_", "rms_"))]
spectral_features = [c for c in df.columns if c.startswith(("centroid_", "rolloff_", "bandwidth_", "flatness_"))]
contrast_features = [c for c in df.columns if c.startswith("contrast_band")]
mfcc_mean = [c for c in df.columns if c.startswith("mfcc") and c.endswith("_mean")]
mfcc_std = [c for c in df.columns if c.startswith("mfcc") and c.endswith("_std")]
mfcc_delta_mean = [c for c in df.columns if c.startswith("mfcc_delta") and c.endswith("_mean")]
mfcc_delta_std = [c for c in df.columns if c.startswith("mfcc_delta") and c.endswith("_std")]
mfcc_dd_mean = [c for c in df.columns if c.startswith("mfcc_delta2") and c.endswith("_mean")]
chroma_features = [c for c in df.columns if c.startswith("chroma")]
tonnetz_features = [c for c in df.columns if c.startswith("tonnetz")]
mel_stats = [c for c in df.columns if c.startswith("mel_db_")]

print("Detected feature groups:")
print(f"  time_features: {time_features}")
print(f"  spectral_features: {spectral_features}")
print(f"  contrast_features: {contrast_features[:7]}{'...' if len(contrast_features)>7 else ''}")
print(f"  mfcc_mean count: {len(mfcc_mean)}")
print(f"  chroma count: {len(chroma_features)}")
print(f"  mel_stats: {mel_stats}")

# ---------- Utility functions ----------
def top_classes(df_labels, n=TOP_N_CLASSES):
    counts = df_labels.value_counts()
    top = counts.nlargest(n).index.tolist()
    return top

def savefig(fig, name, dpi=150):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {path}")

# ---------- 1) Time-domain boxplots (zcr_mean, rms_mean) ----------
time_means = [c for c in time_features if c.endswith("_mean")]
if time_means:
    cols = time_means
    top = top_classes(df['label'])
    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = []
        names = []
        for lbl in top:
            subset = df[df['label'] == lbl][col].dropna()
            data.append(subset.values)
            names.append(lbl)
        ax.boxplot(data, tick_labels=names, vert=True, showfliers=False)
        ax.set_title(col)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=45)
    savefig(fig, "time_domain_boxplots.png")
else:
    print("No time-domain mean features found for boxplots.")

# ---------- 2) Frequency-domain boxplots ----------
freq_means = [c for c in spectral_features if c.endswith("_mean")]
if freq_means:
    cols = freq_means
    top = top_classes(df['label'])
    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        data = []
        names = []
        for lbl in top:
            subset = df[df['label'] == lbl][col].dropna()
            data.append(subset.values)
            names.append(lbl)
        ax.boxplot(data, tick_labels=names, vert=True, showfliers=False)
        ax.set_title(col)
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=45)
    savefig(fig, "frequency_domain_boxplots.png")
else:
    print("No spectral mean features found for boxplots.")

# ---------- 3) Spectral contrast per band (mean) ----------
contrast_mean_cols = [c for c in contrast_features if c.endswith("_mean")]
if contrast_mean_cols:
    top = top_classes(df['label'])
    fig, ax = plt.subplots(figsize=(10, 6))
    band_idx = list(range(1, len(contrast_mean_cols)+1))
    for lbl in top:
        vals = df[df['label'] == lbl][contrast_mean_cols].mean(axis=0).values
        ax.plot(band_idx, vals, marker='o', label=lbl)
    ax.set_xticks(band_idx)
    ax.set_xlabel("Contrast band (1..{})".format(len(contrast_mean_cols)))
    ax.set_ylabel("Mean spectral contrast")
    ax.set_title("Mean spectral contrast per band — top classes")
    ax.legend()
    savefig(fig, "spectral_contrast_by_class.png")
else:
    print("No spectral contrast features found.")

# ---------- 4) Mean MFCC curve for top classes ----------
if mfcc_mean:
    # sort mfcc_mean by index order (mfcc1_mean..mfcc20_mean)
    def mfcc_index(colname):
        # extract number
        import re
        m = re.search(r"mfcc(\d+)_mean", colname)
        return int(m.group(1)) if m else 999
    mfcc_mean_sorted = sorted(mfcc_mean, key=mfcc_index)
    top = top_classes(df['label'])
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(1, len(mfcc_mean_sorted) + 1))
    for lbl in top:
        mean_vec = df[df['label'] == lbl][mfcc_mean_sorted].mean(axis=0).values
        ax.plot(x, mean_vec, marker='o', label=lbl)
    ax.set_xlabel("MFCC index")
    ax.set_xticks(x)
    ax.set_ylabel("Mean MFCC coefficient")
    ax.set_title("Mean MFCC (1..{}) per class — top classes".format(len(mfcc_mean_sorted)))
    ax.legend()
    savefig(fig, "mfcc_mean_by_class.png")
else:
    print("No MFCC mean features found.")

# ---------- 5) Chroma heatmap (avg chroma per class) ----------
if chroma_features:
    top = top_classes(df['label'])
    chroma_sorted = sorted(chroma_features, key=lambda c: int(c.replace("chroma", "").split("_")[0]))
    # build a matrix: rows=classes, cols=chroma bins
    matrix = []
    names = []
    for lbl in top:
        vals = df[df['label'] == lbl][chroma_sorted].mean(axis=0).values
        matrix.append(vals)
        names.append(lbl)
    matrix = np.vstack(matrix)
    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.8)))
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(range(len(chroma_sorted)))
    ax.set_xticklabels([f"C{i+1}" for i in range(len(chroma_sorted))])
    ax.set_title("Average chroma vector per class (top classes)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    savefig(fig, "chroma_heatmap_top_classes.png")
else:
    print("No chroma features found.")

# ---------- 6) Correlation heatmap for representative features ----------
# choose a compact representative set (if present)
rep_candidates = [
    "zcr_mean", "rms_mean", "centroid_mean", "rolloff_mean", "bandwidth_mean", "flatness_mean",
    "mel_db_mean"
]
# add first 5 MFCC means if exist
rep_mfcc = mfcc_mean_sorted[:5] if mfcc_mean else []
rep = [c for c in rep_candidates if c in df.columns] + rep_mfcc
if len(rep) >= 2:
    corr = df[rep].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu")
    ax.set_xticks(range(len(rep)))
    ax.set_xticklabels(rep, rotation=45, ha="right")
    ax.set_yticks(range(len(rep)))
    ax.set_yticklabels(rep)
    ax.set_title("Feature correlation (representative subset)")
    cbar = fig.colorbar(im, ax=ax)
    savefig(fig, "representative_correlation.png")
else:
    print("Not enough representative columns found for correlation heatmap.")

# ---------- 7) PCA scatter (first 2 PCs) colored by class ----------
# build numeric feature matrix excluding 'label' and 'file'
exclude = {'label', 'file'}
num_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
print(f"Using {len(num_cols)} numeric features for PCA")
if len(num_cols) >= 2:
    X = df[num_cols].fillna(0).values
    y = df['label'].astype(str).values

    # sample if too big
    if len(X) > SAMPLE_FOR_PCA:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X), SAMPLE_FOR_PCA, replace=False)
        Xs = X[idx]
        ys = y[idx]
    else:
        Xs = X
        ys = y

    # scale
    scaler = StandardScaler()
    Xs_scaled = scaler.fit_transform(Xs)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(Xs_scaled)

    # plot — color by class, but limit to top classes to avoid color explosion
    top = top_classes(df['label'], n=TOP_N_CLASSES)
    selected_mask = np.isin(ys, top)
    pcs_sel = pcs[selected_mask]
    ys_sel = ys[selected_mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    unique = list(sorted(set(ys_sel)))

    for i, lbl in enumerate(unique):
        mask = (ys_sel == lbl)
        ax.scatter(pcs_sel[mask, 0], pcs_sel[mask, 1], s=10, alpha=0.7, label=lbl)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA (2 components) — top classes")
    ax.legend(fontsize="small", markerscale=2)
    savefig(fig, "pca_scatter_top_classes.png")
else:
    print("Not enough numeric features for PCA.")

# ---------- 8) Quick per-class summary CSV ----------
# produce a small CSV with means of main groups for quick inspection
summary_cols = []
summary_cols += [c for c in time_means]
summary_cols += [c for c in freq_means]
summary_cols += mfcc_mean_sorted[:5]  # first 5 MFCC means
summary_cols += chroma_sorted[:12] if chroma_features else []
summary_cols = [c for c in summary_cols if c in df.columns]
class_summary = df.groupby('label')[summary_cols].mean().reset_index()
class_summary_path = os.path.join(OUT_DIR, "class_summary_means.csv")
class_summary.to_csv(class_summary_path, index=False)
print(f"Saved class summary means to {class_summary_path}")

print("\nAll done. Check the '{}' folder for plots.".format(OUT_DIR))