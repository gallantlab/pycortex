"""
================================================
Inspect Encoding-Model Predictions and Timeseries
================================================

Load a static prediction-performance map (3D) together with the actual and
predicted BOLD responses it was computed from (both 4D) into one WebGL
viewer. Browse the R² map on the surface, open the timeseries panel
(Open Controls -> movie -> timeseries) and click a voxel: the panel plots
the actual and predicted timecourses at that location, so you can see *what*
a voxel responds to and how well the model captures it. Plain 1D arrays in
the data dict (the block design here) appear as optional reference traces.
Clicking a timepoint in the panel seeks the 4D datasets to that volume;
switch to the "actual" dataset (+ / - keys) to watch the brain follow.

The data is synthetic and built for the example subject (S1), so the example
runs without any download. The experiment is a block design of 240 volumes:
20-volume blocks of stimulus 1, stimulus 2 and rest, repeated four times.
Six Gaussian blobs are placed at well-separated random locations inside
the cortical mask, with different tuning at a high and a low signal-to-noise level. 
The "actual" response is the tuned, HRF-convolved block design plus noise; the
"predicted" response is the model's version of it, and the R² map is
computed from the two. Everything outside the blobs is noise.
"""

import numpy as np
from scipy.stats import gamma
import cortex

def pick_centers(candidates, n, min_dist):
    """Random candidate voxels at least `min_dist` apart; 
    the spacing is relaxed if the candidate set cannot host n such points."""
    chosen = []
    while len(chosen) < n:
        for i in np.random.permutation(len(candidates)):
            c = candidates[i]
            if all(np.linalg.norm(c - k) >= min_dist for k in chosen):
                chosen.append(c)
                if len(chosen) == n:
                    break
        else:
            min_dist *= 0.8
    return np.array(chosen)

def blob(center, radius):
    d2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return np.exp(-d2 / (2 * radius ** 2))


np.random.seed(42)
subject, xfmname = "S1", "fullhead"
xfm = cortex.db.get_xfm(subject, xfmname)
shape = xfm.shape                  # (z, y, x) = (31, 100, 100)

# --- block design: stim1 (20) - stim2 (20) - rest (20), x4 = 240 volumes
block, n_cycles = 20, 4
n_volumes = 3 * block * n_cycles
stim1 = np.tile(np.r_[np.ones(block), np.zeros(2 * block)], n_cycles)
stim2 = np.tile(np.r_[np.zeros(block), np.ones(block), np.zeros(block)], n_cycles)

# --- BOLD regressors: convolved with a canonical HRF (TR = 2 s)
tr = 2.0
t_hrf = np.arange(0, 32, tr)
hrf = gamma.pdf(t_hrf, 6) - 0.35 * gamma.pdf(t_hrf, 12)
hrf /= hrf.sum()
reg1 = np.convolve(stim1, hrf)[:n_volumes]
reg2 = np.convolve(stim2, hrf)[:n_volumes]

# --- six blobs centered on cortical voxels
mask = cortex.utils.get_cortical_mask(subject, xfmname, type="nearest")  # (z, y, x)
gray = np.argwhere(mask)

tuning = [(1.0, 0.2),    # blob 1: stimulus 1 (high SNR)
          (0.2, 1.0),    # blob 2: stimulus 2 (high SNR)
          (1.0, 1.0),    # blob 3: both (high SNR)
          (0.6, 0.1),    # blob 4: stimulus 1 (low SNR)
          (0.1, 0.6),    # blob 5: stimulus 2 (low SNR)
          (0.6, 0.6)]    # blob 6: both (low SNR)

n_blobs, radius = len(tuning), 6.0

centers = pick_centers(gray, n_blobs, min_dist=4*radius)
zz, yy, xx = np.mgrid[:shape[0], :shape[1], :shape[2]]

# --- actual and predicted responses
predicted = np.zeros((n_volumes,) + shape)
for center, (a1, a2) in zip(centers, tuning):
    predicted += blob(center, radius) * (a1 * reg1 + a2 * reg2)[:, None, None, None]
actual = predicted + 0.4 * np.random.randn(n_volumes, *shape)
predicted += 0.2 * np.random.randn(n_volumes, *shape)

# --- prediction performance per voxel
r2 = 1 - ((actual - predicted) ** 2).sum(0) / ((actual - actual.mean(0)) ** 2).sum(0)
r2 = np.clip(r2, 0, 1)
r2[r2<=0.0] = np.nan # thresholding

volumes = {
    "R2": cortex.Volume(r2, subject, xfmname,
                                 cmap="hot", vmin=0.0, vmax=1.0),
    "actual": cortex.Volume(actual, subject, xfmname,
                            vmin=-1.5, vmax=1.5),
    "predicted": cortex.Volume(predicted, subject, xfmname,
                               vmin=-1.5, vmax=1.5),
    "stimulus 1": stim1,        # block design, shown as reference traces
    "stimulus 2": stim2,
}

cortex.webshow(volumes)
print("blob centers (z, y, x):", centers.tolist())
# Like the other WebGL examples, run this from ipython/jupyter or with
# `python -i` so the interpreter (and the viewer's server) stays alive.
