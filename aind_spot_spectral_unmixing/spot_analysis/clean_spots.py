
# clean_spot_codex.py
"""
Minimal, KDTree-based pipeline for selecting clean 3D spots and estimating
a crosstalk matrix. Channel count is auto-detected from columns named
'chan_{i}_intensity' and can range from 2 to 6 (or more if present).

Requirements:
  - numpy, pandas
  - scipy (optional but recommended): uses scipy.spatial.cKDTree for speed.
    If SciPy is unavailable, falls back to a slower NumPy implementation.

Data assumptions:
  - DataFrame columns: 'x','y','z','r' (optional 'r' threshold), and channel columns
    like 'chan_0_intensity', 'chan_1_intensity', ..., 'chan_{k-1}_intensity'.
  - Optionally includes grouping columns such as 'round' and 'fov' to enforce
    isolation within the same round/FOV.

Clean-spot criteria (no 3D shell image needed):
  1) 3D isolation: nearest neighbor distance (across all channels) must exceed R_iso.
  2) Neighbor density: count of neighbors within [r_in, r_out] must be <= max_neighbors.
  3) Channel purity: dominant channel fraction >= min_top_frac and
                     top/second ratio >= min_top_to_second.
  4) Fit quality (optional): r >= r_min, if 'r' provided.

Crosstalk estimation:
  - Uses k-means on unit-normalized channel vectors to learn per-dye profiles.
  - Number of clusters (k) = number of detected channel columns.
  - Returns both unit-norm centroids and a row-normalized crosstalk matrix.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd

try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_SCIPY = True
except Exception:
    KDTree = None
    _HAVE_SCIPY = False

# ----------------------- Utilities -----------------------

def _channel_cols(df: pd.DataFrame, prefix: str = "chan_", suffix: str = "_intensity") -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
    # sort numerically if possible
    def _key(c):
        try:
            return int(c[len(prefix): -len(suffix)])
        except Exception:
            return c
    return sorted(cols, key=_key)

def _anisotropic_scale(coords: np.ndarray, voxel_size_xyz: Tuple[float, float, float]) -> np.ndarray:
    sx, sy, sz = voxel_size_xyz
    S = np.array([sx, sy, sz], dtype=float)
    return coords * S

def _nn_distance_and_counts(coords_scaled: np.ndarray, r_in: float, r_out: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (d_nn, n_in_shell) for each point, where:
      - d_nn is the distance to the nearest other point
      - n_in_shell is the number of neighbors with distance in (r_in, r_out]
    Uses KDTree when available; else NumPy fallback.
    """
    N = coords_scaled.shape[0]
    if N == 0:
        return np.array([]), np.array([])

    if _HAVE_SCIPY:
        tree = KDTree(coords_scaled)
        # nearest neighbor distance (k=2 includes self)
        dists, idxs = tree.query(coords_scaled, k=2, eps=0.0, workers=-1)
        d_nn = dists[:, 1]
        # neighbor counts in [r_in, r_out]: compute counts within r_out minus counts within r_in
        n_out = tree.query_ball_point(coords_scaled, r_out, workers=-1)
        if r_in > 0:
            n_in  = tree.query_ball_point(coords_scaled, r_in, workers=-1)
            n_shell = np.array([len(n_out[i]) - len(n_in[i]) for i in range(N)], dtype=int)
        else:
            n_shell = np.array([len(lst) - 1 for lst in n_out], dtype=int)  # minus self
        return d_nn, n_shell

    # Fallback: NumPy brute force (O(N^2)), chunked to reduce memory blowup.
    d_nn = np.full(N, np.inf, dtype=float)
    n_shell = np.zeros(N, dtype=int)
    chunk = 4096
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        A = coords_scaled[i0:i1]  # (M, 3)
        # squared distances to all
        A2 = np.sum(A**2, axis=1, keepdims=True)         # (M,1)
        B2 = np.sum(coords_scaled**2, axis=1, keepdims=True).T  # (1,N)
        D2 = A2 + B2 - 2.0 * (A @ coords_scaled.T)       # (M,N)
        # set self distances to inf within the sub-block
        for i in range(i1 - i0):
            D2[i, i + i0] = np.inf
        D = np.sqrt(D2, where=np.isfinite(D2), out=np.full_like(D2, np.inf))
        d_nn[i0:i1] = np.min(D, axis=1)
        in_shell = (D > r_in) & (D <= r_out)
        n_shell[i0:i1] = in_shell.sum(axis=1)
    return d_nn, n_shell

def _unit_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / norms

def _kmeans_unit(X: np.ndarray, k: int, n_init: int = 8, max_iter: int = 150, seed: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means on unit-normalized rows. Returns (centroids_unit, labels).
    Light k-means++ init; no external deps beyond NumPy.
    """
    rng = np.random.default_rng(seed)
    Xu = _unit_normalize_rows(X)
    N, D = Xu.shape

    best_inertia = np.inf
    best_C = None
    best_labels = None

    def _init_plus_plus():
        C = np.empty((k, D), dtype=float)
        C[0] = Xu[rng.integers(N)]
        # squared distances to nearest chosen center
        D2 = np.sum((Xu - C[0])**2, axis=1)
        for ci in range(1, k):
            probs = D2 / (D2.sum() + 1e-12)
            idx = rng.choice(N, p=probs)
            C[ci] = Xu[idx]
            D2 = np.minimum(D2, np.sum((Xu - C[ci])**2, axis=1))
        return C

    for _ in range(n_init):
        C = _init_plus_plus()
        for _it in range(max_iter):
            # assign
            d2 = np.sum(Xu**2, axis=1, keepdims=True) + np.sum(C**2, axis=1) - 2 * (Xu @ C.T)
            labels = np.argmin(d2, axis=1)
            # update
            C_new = np.vstack([Xu[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
            C_new = _unit_normalize_rows(C_new)
            if np.allclose(C_new, C, atol=1e-6):
                C = C_new
                break
            C = C_new
        inertia = np.sum([np.sum((Xu[labels == j] - C[j])**2) for j in range(k)])
        if inertia < best_inertia:
            best_inertia = inertia
            best_C = C.copy()
            best_labels = labels.copy()
    return best_C, best_labels
"""
Estimate a slopes-based mixing matrix by fitting "dyelines":
  For each dye (cluster), pick a reference channel (the channel with the highest
  mean intensity for that dye). For each channel j, fit a line through the origin
  y_j ≈ beta_j * y_ref across all spots in that dye, using ordinary least squares.
  The beta_j values form the dye's row in the slopes matrix.

Orientation:
  - Returns matrix S of shape (k_dyes, k_channels), where S[d, j] is the slope
    for channel j for dye d. This is a *mixing* matrix in the same orientation
    as previous `M` (rows = dyes, cols = channels).

Robustness:
  - By default, uses all clean spots for each dye. Optionally supports
    outlier trimming via percentile clipping.

Inputs:
  - df: DataFrame with columns ['x','y','z'] (not used here) and channel columns
        named 'chan_{i}_intensity'.
  - labels: 1D array-like of dye labels (length == len(df)). If None, will try
            `df['dye']` and otherwise raise.

Returns:
  dict with:
    - 'slopes': (k, C) array
    - 'ref_channels': list of length k (chosen reference channel for each dye)
    - 'channels': the list of channel column names used
    - 'means_by_dye': (k, C) mean intensities per dye
"""
from typing import Optional, List, Dict
import numpy as np
import pandas as pd

def _channel_cols(df: pd.DataFrame, prefix: str = "chan_", suffix: str = "_intensity") -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix) and c.endswith(suffix)]
    def _key(c):
        try:
            return int(c[len(prefix): -len(suffix)])
        except Exception:
            return c
    return sorted(cols, key=_key)

def _ols_slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    # beta = (x^T y) / (x^T x)
    num = float(np.dot(x, y))
    den = float(np.dot(x, x)) + 1e-12
    return max(0.0, num / den)  # enforce non-negativity

def estimate_dyeline_slopes(
    df: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    channel_cols: Optional[List[str]] = None,
    trim_pct: float = 0.0   # e.g., 1.0 trims 1% low/high of ref channel within each dye
) -> Dict[str, np.ndarray]:
    if channel_cols is None:
        channel_cols = _channel_cols(df)
    if len(channel_cols) < 2:
        raise ValueError("Need at least 2 channel columns named like 'chan_{i}_intensity'.")

    if labels is None:
        if 'dye' in df.columns:
            labels = df['dye'].to_numpy()
        else:
            raise ValueError("labels must be provided, or df must contain a 'dye' column.")
    labels = np.asarray(labels).astype(int)

    C = len(channel_cols)
    k = int(labels.max()) + 1  # assumes labels are 0..k-1
    X = df[channel_cols].to_numpy(float)
    X = np.clip(X, 0, None)

    # Compute per-dye channel means and choose reference channel = argmax mean
    means_by_dye = np.zeros((k, C), dtype=float)
    ref_channels = np.zeros(k, dtype=int)
    slopes = np.zeros((k, C), dtype=float)

    for d in range(k):
        idx = np.where(labels == d)[0]
        if idx.size == 0:
            continue
        Xd = X[idx]  # (Nd, C)
        means_by_dye[d] = Xd.mean(axis=0)
        ref = int(np.argmax(means_by_dye[d]))
        ref_channels[d] = ref

        x_ref = Xd[:, ref]

        # Optional trimming on x_ref to reduce outliers
        if trim_pct > 0.0:
            lo = np.percentile(x_ref, trim_pct)
            hi = np.percentile(x_ref, 100.0 - trim_pct)
            keep = (x_ref >= lo) & (x_ref <= hi)
            Xd = Xd[keep]
            x_ref = Xd[:, ref]

        # Fit slopes y_j ~ beta_j * x_ref for all channels
        # Vectorized: beta = (x_ref^T Y) / (x_ref^T x_ref)
        num = x_ref.T @ Xd  # shape (C,)
        den = float(x_ref.T @ x_ref) + 1e-12
        beta = num / den
        beta = np.clip(beta, 0.0, None)  # enforce non-negativity
        slopes[d] = beta

        # (Optional) Normalize so the reference channel's slope is exactly 1.0
        if beta[ref] > 0:
            slopes[d] /= beta[ref]

    return {
        "slopes": slopes,
        "ref_channels": ref_channels,
        "channels": channel_cols,
        "means_by_dye": means_by_dye,
    }

# ----------------------- Main API -----------------------

def compute_clean_mask_kdtree(
    df: pd.DataFrame,
    voxel_size_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    R_iso: float = 6.0,
    r_in: float = 4.0,
    r_out: float = 14.0,
    max_neighbors: int = 0,
    min_top_frac: float = 0.7,
    min_top_to_second: float = 2.0,
    r_min: Optional[float] = None,
    group_cols: Optional[List[str]] = None
) -> pd.Series:
    """
    Return boolean mask of clean spots using 3D isolation + neighbor density + optional r-threshold.
    Distances operate on anisotropically scaled coordinates given voxel_size_xyz.
    Neighbor density counts neighbors in (r_in, r_out].

    Parameters:
    - df: DataFrame with columns 'x','y','z','r' (optional) and channel columns like 'chan_{i}_intensity'.
    - voxel_size_xyz: tuple of (sx, sy, sz) voxel sizes to scale coordinates.
    - R_iso: minimum nearest neighbor distance to consider a spot isolated. Set to None to skip isolation check.
    - r_in, r_out: inner and outer radii for neighbor counting.
    - max_neighbors: maximum allowed neighbors in the shell (r_in, r_out].
    - min_top_frac: minimum fraction of total intensity in the top channel.
    - min_top_to_second: minimum ratio of top channel intensity to second channel intensity.
    - r_min: optional minimum 'r' value (if 'r' column present) for fit quality.
    - group_cols: optional list of columns to group by (e.g., ['round', 'fov']) so that
                  isolation and density are computed within each group separately.
                  If None, will use ['round', 'round_id', 'fov'] if present in df.
    Returns:
    - pd.Series of booleans, index aligned with df, True = clean spot.
    """
    if group_cols is None:
        group_cols = [c for c in ("round", "round_id", "fov") if c in df.columns]
    ch_cols = _channel_cols(df)
    if len(ch_cols) < 2:
        raise ValueError("Need at least 2 channel columns named like 'chan_{i}_intensity'.")

    clean = pd.Series(False, index=df.index)

    # Precompute purity terms globally (used inside groups via row indexing)
    X = df[ch_cols].to_numpy(float)
    X = np.clip(X, 0, None)
    row_sum = X.sum(axis=1) + 1e-12
    top_idx = np.argmax(X, axis=1)
    top_val = X[np.arange(X.shape[0]), top_idx]
    second_val = np.partition(X, -2, axis=1)[:, -2]
    top_frac = top_val / row_sum
    top_ratio = (top_val + 1e-12) / (second_val + 1e-12)

    r_ok = np.ones(len(df), dtype=bool)
    if r_min is not None and "r" in df.columns:
        r_ok = df["r"].to_numpy(float) >= r_min

    # Group or not
    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = [(None, df)]

    for _, g in groups:
        idx = g.index
        coords = g[["x","y","z"]].to_numpy(float)
        coords_scaled = _anisotropic_scale(coords, voxel_size_xyz)
        d_nn, n_shell = _nn_distance_and_counts(coords_scaled, r_in=r_in, r_out=r_out)

        iso_ok = np.ones(len(g), dtype=bool) if R_iso is None else (d_nn > R_iso) # allow skip isolation

        dens_ok  = n_shell <= max_neighbors
        pur_ok   = (top_frac[idx] >= min_top_frac) & (top_ratio[idx] >= min_top_to_second)
        group_ok = iso_ok & dens_ok & pur_ok & r_ok[idx]
        clean.loc[idx] = group_ok

    return clean

def estimate_crosstalk_matrix(
    df_clean: pd.DataFrame,
    channel_cols: Optional[List[str]] = None,
    seed: Optional[int] = 0,
    n_init: int = 8,
    max_iter: int = 150
) -> Dict[str, np.ndarray]:
    """
    Estimate crosstalk with k = number of channel columns (auto: 2–6+).
    Returns dict with 'centroids' (unit-norm) and 'matrix' (rows sum to 1), plus 'labels'.
    """
    if channel_cols is None:
        channel_cols = _channel_cols(df_clean)
    k = len(channel_cols)
    if k < 2:
        raise ValueError("At least 2 channels required to estimate crosstalk.")

    X = df_clean[channel_cols].to_numpy(float)
    X = np.clip(X, 0, None) + 1e-12
    C_unit, labels = _kmeans_unit(X, k=k, n_init=n_init, max_iter=max_iter, seed=seed)
    # Row-normalize (non-negative) to get a proper crosstalk matrix
    M = np.clip(C_unit, 0, None)
    M = M / (M.sum(axis=1, keepdims=True) + 1e-12)
    return {"centroids": C_unit, "matrix": M, "labels": labels, "channels": channel_cols}

def select_clean_and_estimate(
    df: pd.DataFrame,
    voxel_size_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    R_iso: float = 6.0,
    r_in: float = 4.0,
    r_out: float = 14.0,
    max_neighbors: int = 0,
    min_top_frac: float = 0.7,
    min_top_to_second: float = 2.0,
    r_min: Optional[float] = None,
    group_cols: Optional[List[str]] = None,
    seed: Optional[int] = 0
) -> Dict[str, object]:
    """
    One-call convenience:
      1) compute clean mask (KDTree if available),
      2) estimate crosstalk matrix from clean spots (k = #channels).
    """
    mask = compute_clean_mask_kdtree(
        df=df,
        voxel_size_xyz=voxel_size_xyz,
        R_iso=R_iso,
        r_in=r_in,
        r_out=r_out,
        max_neighbors=max_neighbors,
        min_top_frac=min_top_frac,
        min_top_to_second=min_top_to_second,
        r_min=r_min,
        group_cols=group_cols,
    )
    df_clean = df[mask].copy()
    xt = estimate_crosstalk_matrix(df_clean, seed=seed)
    return {"clean_mask": mask, "df_clean": df_clean, **xt}

def predict_gene_profile(
    gene_dyes_by_round: Dict[int, int],
    crosstalk_matrix: np.ndarray,
    n_rounds: int
) -> np.ndarray:
    """
    Concatenate per-round channel profiles using assigned dye per round.
    crosstalk_matrix has shape (k_channels, k_channels); row i is profile for dye i.
    """
    profiles = []
    for r in range(n_rounds):
        dye = int(gene_dyes_by_round.get(r, 0))
        profiles.append(crosstalk_matrix[dye])
    return np.concatenate(profiles, axis=0)


# --- Plot dye lines
import numpy as np
import matplotlib.pyplot as plt

def pair_dyeline_slope(S, ci, cj, eps=1e-12):
    """
    Convert dyeline slopes (each dye relative to its own ref channel) into
    a slope for channel cj vs ci: m_d = S[d, cj] / S[d, ci].
    Returns array shape (k_dyes,).
    """
    S = np.asarray(S, dtype=float)
    return S[:, cj] / (S[:, ci] + eps)

def plot_pair_with_dyelines(
    df,
    ch_x_name, ch_y_name,            # e.g. 'chan_488_intensity', 'chan_514_intensity'
    labels,                           # 1D array-like dye labels (0..k-1) for df rows
    S, channels,                      # slopes matrix (k x C) and the ordered list of channel column names
    max_points=5000,                  # downsample for speed/clarity
    log_scale=True,                   # log axes often help for intensities
    quantile_limits=(1.0, 99.5),      # robust axis limits from data
    alpha=0.3                         # point transparency
):
    # Map channel names to indices
    ch_to_idx = {name: i for i, name in enumerate(channels)}
    if ch_x_name not in ch_to_idx or ch_y_name not in ch_to_idx:
        raise ValueError("ch_x_name / ch_y_name not found in channels list.")
    ci, cj = ch_to_idx[ch_x_name], ch_to_idx[ch_y_name]

    # Data for the two channels
    x = df[ch_x_name].to_numpy(float)
    y = df[ch_y_name].to_numpy(float)
    labels = np.asarray(labels).astype(int)
    k = int(labels.max()) + 1

    # Optional downsampling for plotting
    if len(x) > max_points:
        idx = np.random.default_rng(0).choice(len(x), size=max_points, replace=False)
        x_plot, y_plot, lab_plot = x[idx], y[idx], labels[idx]
    else:
        x_plot, y_plot, lab_plot = x, y, labels

    # Compute dyeline slopes for this channel pair
    m = pair_dyeline_slope(S, ci, cj)  # shape (k,)

    # Build robust axis limits from quantiles
    def _lims(v):
        qlo, qhi = np.nanpercentile(v[v > 0], quantile_limits) if np.any(v > 0) else (0, 1)
        if not np.isfinite(qlo): qlo = 0
        if not np.isfinite(qhi): qhi = max(1.0, np.nanmax(v))
        if qlo <= 0: qlo = max(1e-3, qhi * 1e-4)  # avoid zero on log
        return qlo, qhi

    xlo, xhi = _lims(x_plot)
    ylo, yhi = _lims(y_plot)

    # Make the figure (single chart, no subplots)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter, colored by dye label (matplotlib default color cycle)
    for d in range(k):
        sel = (lab_plot == d)
        if not np.any(sel):
            continue
        ax.scatter(x_plot[sel], y_plot[sel], s=6, alpha=alpha, label=f"dye {d}")

    # Overlay dyeline per dye: y = m_d * x, plotted across current x-range
    x_line = np.linspace(xlo, xhi, 200)
    for d in range(k):
        y_line = m[d] * x_line
        ax.plot(x_line, y_line, linewidth=2, label=f"dyeline {d} (slope={m[d]:.2f})")

    ax.set_xlabel(ch_x_name)
    ax.set_ylabel(ch_y_name)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)

    # Combine legends (points + lines)
    ax.legend(loc="best", fontsize=8, ncol=1)
    ax.set_title(f"Pairwise intensities with dyelines: {ch_y_name} vs {ch_x_name}")
    fig.tight_layout()
    return fig, ax


# ----------------------- Neighbor counts utilities -----------------------

# ----------------------- Neighbor counts utilities -----------------------
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

def infer_top_channel_labels(df: pd.DataFrame, channel_cols: Optional[List[str]] = None):
    """
    Return two Series (aligned to df.index):
      - top_channel_idx: int index of the strongest channel (argmax over channel_cols)
      - top_channel_name: str, the corresponding channel column name
    """
    if channel_cols is None:
        channel_cols = _channel_cols(df)
    X = np.clip(df[channel_cols].to_numpy(float), 0, None)
    idx = np.argmax(X, axis=1)
    names = np.array(channel_cols, dtype=object)[idx]
    return pd.Series(idx, index=df.index, name="top_channel_idx"), pd.Series(names, index=df.index, name="top_channel_name")

def compute_neighbor_counts_per_spot(
    df: pd.DataFrame,
    voxel_size_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    r_in: float = 0.0,
    r_out: float = 2.0,
    group_cols: Optional[List[str]] = None,
    include_nn: bool = True,
    channel_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute, for each spot, the number of neighbors with distance in (r_in, r_out]
    using anisotropically scaled coordinates (µm if voxel_size_xyz is in µm/px).

    Returns a DataFrame with:
      - 'n_neighbors_shell': neighbor count in (r_in, r_out]
      - 'd_nn' (optional): nearest-neighbor distance
      - 'top_channel_idx', 'top_channel_name': channel label for the spot
      - if df contains a 'channel' column, it is copied through
    Counts consider ALL spots within each group (e.g., within the same round/FOV).
    """
    if channel_cols is None:
        channel_cols = _channel_cols(df)
    if group_cols is None:
        group_cols = [c for c in ("round", "round_id", "fov") if c in df.columns]

    # Precompute channel labels
    top_idx_s, top_name_s = infer_top_channel_labels(df, channel_cols=channel_cols)

    # Prepare outputs
    out = pd.DataFrame(index=df.index)
    out["top_channel_idx"] = top_idx_s
    out["top_channel_name"] = top_name_s
    if "channel" in df.columns:
        out["channel"] = df["channel"]

    # Grouping
    groups = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]

    all_dnn = []
    all_nshell = []
    for _, g in groups:
        idx = g.index
        coords = g[["x","y","z"]].to_numpy(float)
        coords_scaled = _anisotropic_scale(coords, voxel_size_xyz)
        d_nn, n_shell = _nn_distance_and_counts(coords_scaled, r_in=r_in, r_out=r_out)
        # stash
        all_nshell.append(pd.Series(n_shell, index=idx))
        if include_nn:
            all_dnn.append(pd.Series(d_nn, index=idx))

    out["n_neighbors_shell"] = pd.concat(all_nshell).sort_index()
    if include_nn:
        out["d_nn"] = pd.concat(all_dnn).sort_index()

    return out

# ----------------------- Neighbor counts (uses 'chan' column) -----------------------
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

def compute_neighbor_counts_per_spot_chan(
    df: pd.DataFrame,
    voxel_size_xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    r_in: float = 0.0,
    r_out: float = 2.0,
    group_cols: Optional[List[str]] = None,
    include_nn: bool = True
) -> pd.DataFrame:
    """
    Compute per-spot neighbor counts in the shell (r_in, r_out] using anisotropically
    scaled coordinates. Uses the existing 'chan' column in df (no inference).

    Inputs:
      - df: must contain columns ['x','y','z','chan'].
      - voxel_size_xyz: scaling applied to coords before distance calcs (e.g., (0.24,0.24,1.0) µm/px).
      - r_in, r_out: shell radii in same units as scaled coords (µm if voxel_size_xyz in µm/px).
      - group_cols: optional list (e.g., ['round','fov']) to restrict neighbor searches within groups.
      - include_nn: if True, also returns nearest-neighbor distance 'd_nn'.

    Returns:
      DataFrame aligned to df.index with:
        - 'chan' (copied from df)
        - 'n_neighbors_shell' (int)
        - 'd_nn' (float, if include_nn=True)
    """
    if "chan" not in df.columns:
        raise ValueError("df must contain a 'chan' column (per-spot channel).")
    if group_cols is None:
        group_cols = [c for c in ("round", "round_id", "fov") if c in df.columns]

    out = pd.DataFrame(index=df.index)
    out["chan"] = df["chan"]

    groups = df.groupby(group_cols, sort=False) if group_cols else [(None, df)]

    parts_counts = []
    parts_dnn = []
    for _, g in groups:
        idx = g.index
        coords = g[["x","y","z"]].to_numpy(float)
        coords_scaled = _anisotropic_scale(coords, voxel_size_xyz)
        d_nn, n_shell = _nn_distance_and_counts(coords_scaled, r_in=r_in, r_out=r_out)
        parts_counts.append(pd.Series(n_shell, index=idx))
        if include_nn:
            parts_dnn.append(pd.Series(d_nn, index=idx))

    out["n_neighbors_shell"] = pd.concat(parts_counts).sort_index()
    if include_nn:
        out["d_nn"] = pd.concat(parts_dnn).sort_index()
    return out



### ---- measure to reference
import numpy as np

def compare_matrices(M_opt: np.ndarray, M_ref: np.ndarray):
    """
    Compare an optimized mixing/unmixing matrix against a reference.

    Returns a dict with:
      - mse: mean squared error over all entries
      - mae: mean absolute error over all entries
      - fro_norm_diff: ||M_opt - M_ref||_F
      - fro_norm_ref:  ||M_ref||_F
      - fro_norm_rel:  ||M_opt - M_ref||_F / (||M_ref||_F + 1e-12)
      - pearson: Pearson correlation of flattened matrices (scale-sensitive)
      - spearman: Spearman rank correlation of flattened matrices (scale-invariant ranks)
      - cosine_sim: cosine similarity between flattened matrices
      - diag_mse: MSE on diagonal entries only
      - offdiag_mse: MSE on off-diagonal entries only
      - diag_mae: MAE on diagonal entries only
      - offdiag_mae: MAE on off-diagonal entries only
    """
    M_opt = np.asarray(M_opt, dtype=np.float64)
    M_ref = np.asarray(M_ref, dtype=np.float64)
    if M_opt.shape != M_ref.shape:
        raise ValueError(f"Shape mismatch: {M_opt.shape} vs {M_ref.shape}")

    diff = M_opt - M_ref
    n = diff.size

    # Basic errors
    mse = np.mean(diff**2)
    mae = np.mean(np.abs(diff))
    fro_diff = np.linalg.norm(diff, "fro")
    fro_ref  = np.linalg.norm(M_ref, "fro")
    fro_rel  = fro_diff / (fro_ref + 1e-12)

    # Flatten + finite mask (avoid NaN/Inf messing with correlations)
    a = M_opt.ravel()
    b = M_ref.ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]

    # Pearson correlation
    def _pearson(x, y):
        if x.size < 2:
            return np.nan
        xz = x - x.mean()
        yz = y - y.mean()
        denom = (np.linalg.norm(xz) * np.linalg.norm(yz)) + 1e-12
        return float(np.dot(xz, yz) / denom)

    pearson = _pearson(a, b)

    # Spearman correlation (rank-based). No SciPy required.
    def _ranks(x):
        # average ranks for ties
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, x.size + 1)
        # handle ties: average ranks among equal values
        # find runs of equal values in sorted order
        xs = x[order]
        i = 0
        while i < xs.size:
            j = i + 1
            while j < xs.size and xs[j] == xs[i]:
                j += 1
            # average rank for the tie block [i, j)
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
            i = j
        return ranks

    if a.size >= 2:
        ra = _ranks(a)
        rb = _ranks(b)
        spearman = _pearson(ra, rb)
    else:
        spearman = np.nan

    # Cosine similarity
    def _cosine(x, y):
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
        return float(np.dot(x, y) / denom)
    cosine_sim = _cosine(a, b)

    # Diagonal vs off-diagonal errors
    if M_opt.shape[0] == M_opt.shape[1]:
        diag_mask = np.eye(M_opt.shape[0], dtype=bool)
        off_mask  = ~diag_mask

        diag_mse = np.mean((M_opt[diag_mask] - M_ref[diag_mask])**2) if diag_mask.any() else np.nan
        off_mse  = np.mean((M_opt[off_mask]  - M_ref[off_mask])**2)  if off_mask.any()  else np.nan

        diag_mae = np.mean(np.abs(M_opt[diag_mask] - M_ref[diag_mask])) if diag_mask.any() else np.nan
        off_mae  = np.mean(np.abs(M_opt[off_mask]  - M_ref[off_mask]))  if off_mask.any()  else np.nan
    else:
        diag_mse = off_mse = diag_mae = off_mae = np.nan

    return {
        "mse": mse,
        "mae": mae,
        "fro_norm_diff": fro_diff,
        "fro_norm_ref": fro_ref,
        "fro_norm_rel": fro_rel,
        "pearson": pearson,
        "spearman": spearman,
        "cosine_sim": cosine_sim,
        "diag_mse": diag_mse,
        "offdiag_mse": off_mse,
        "diag_mae": diag_mae,
        "offdiag_mae": off_mae,
        "n_elements": int(n),
    }



def offdiag_error_by_column(M_opt: np.ndarray, M_ref: np.ndarray):
    """
    Compute per-column off-diagonal errors between two same-shaped matrices.
    Returns:
      - overall_offdiag_mse
      - overall_offdiag_mae
      - per_col_offdiag_mse: (C,) array, MSE for off-diagonals in each column
      - per_col_offdiag_mae: (C,) array, MAE for off-diagonals in each column
    """
    M_opt = np.asarray(M_opt, dtype=np.float64)
    M_ref = np.asarray(M_ref, dtype=np.float64)

    if M_opt.shape != M_ref.shape:
        raise ValueError(f"Shape mismatch: {M_opt.shape} vs {M_ref.shape}")
    if M_opt.shape[0] != M_opt.shape[1]:
        raise ValueError("This breakdown expects a square matrix (same # rows/cols).")

    C = M_opt.shape[0]
    diff = M_opt - M_ref

    # Masks
    eye = np.eye(C, dtype=bool)
    off_mask = ~eye

    # Overall off-diagonal errors
    off_vals = diff[off_mask]
    overall_offdiag_mse = float(np.mean(off_vals**2)) if off_vals.size else np.nan
    overall_offdiag_mae = float(np.mean(np.abs(off_vals))) if off_vals.size else np.nan

    # Per-column: exclude the diagonal element for that column
    per_col_offdiag_mse = np.empty(C, dtype=np.float64)
    per_col_offdiag_mae = np.empty(C, dtype=np.float64)

    for j in range(C):
        col_off = diff[:, j].copy()
        col_off[j] = np.nan  # ignore diagonal
        valid = ~np.isnan(col_off)
        if valid.any():
            v = col_off[valid]
            per_col_offdiag_mse[j] = np.mean(v**2)
            per_col_offdiag_mae[j] = np.mean(np.abs(v))
        else:
            per_col_offdiag_mse[j] = np.nan
            per_col_offdiag_mae[j] = np.nan

    return {
        "overall_offdiag_mse": overall_offdiag_mse,
        "overall_offdiag_mae": overall_offdiag_mae,
        "per_col_offdiag_mse": per_col_offdiag_mse,
        "per_col_offdiag_mae": per_col_offdiag_mae,
    }
