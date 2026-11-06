"""Time series, sliding-window embeddings, and distance-matrix utilities."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def sliding_window(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    window: int,
) -> pd.DataFrame:
    """Embed a time series with a sliding window.

    Each row of ``data`` is an observation. The output has one row per window
    position and ``window * p`` columns, where ``p`` is the number of variables.

    Parameters
    ----------
    data : DataFrame, Series, ndarray, or list
        The input time series.
    window : int
        The sliding-window length, between 1 and ``len(data)``.

    Returns
    -------
    pandas.DataFrame
        The sliding-window embedding.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data) if data.ndim == 1 else pd.DataFrame(data)
    elif isinstance(data, list):
        data = pd.Series(data)
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("data must be a DataFrame, Series, ndarray, or list.")
    if not isinstance(window, int) or window < 1 or window > len(data):
        raise ValueError("window must be an integer in [1, len(data)].")

    rows = range(len(data) - window + 1)
    if isinstance(data, pd.DataFrame):
        out = [data.iloc[i : i + window].values.flatten() for i in rows]
    else:
        out = [data.iloc[i : i + window].values for i in rows]
    return pd.DataFrame(out)


def sw_dist_mats(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list],
    window: int = 1,
    metric: str = "manhattan",
    glob: bool = False,
) -> np.ndarray:
    """Compute sliding-window distance matrices.

    Parameters
    ----------
    data : DataFrame, Series, ndarray, or list
        The input time series, with observations in rows and variables in columns.
    window : int, optional
        The sliding-window length.
    metric : str, optional
        A metric understood by :func:`sklearn.metrics.pairwise_distances`.
    glob : bool, optional
        If ``True``, treat ``data`` as a single multivariate series and return one
        distance matrix. If ``False`` (default), return one matrix per variable.

    Returns
    -------
    numpy.ndarray
        Upper-triangular distance matrices: one of shape ``(m, m)`` if ``glob``
        is ``True``, otherwise an array of shape ``(p, m, m)`` with one matrix
        per variable, where ``m`` is the number of window positions.
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data) if data.ndim == 1 else pd.DataFrame(data)
    elif isinstance(data, list):
        data = pd.Series(data)

    if glob:
        emb = sliding_window(data, window)
        return np.triu(pairwise_distances(emb, metric=metric), k=1)

    if isinstance(data, pd.Series):
        data = data.to_frame()
    mats = [
        np.triu(pairwise_distances(sliding_window(data[col], window), metric=metric), k=1)
        for col in data.columns
    ]
    return np.array(mats)


def weighted_matrix_sum(
    matrices: Union[list, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Return the weighted sum ``sum_j weights[j] * matrices[j]``."""
    return np.einsum("i,ijk->jk", weights, matrices)


def extract_vectors(matrices: np.ndarray, indices: list) -> np.ndarray:
    """Stack the entries of each matrix at the given ``(row, col)`` positions.

    Parameters
    ----------
    matrices : numpy.ndarray
        An array of shape ``(l, m, m)``.
    indices : array-like
        The ``(row, col)`` positions, one per row.

    Returns
    -------
    numpy.ndarray
        An array of shape ``(len(indices), l)``.
    """
    out = np.empty((len(indices), matrices.shape[0]), dtype=matrices.dtype)
    for i, (p, q) in enumerate(indices):
        out[i] = matrices[:, p, q]
    return out


def generate_three_sine_dataset(
    n_points: int = 300,
    period: int = 50,
    noise_sds=(1, 2, 3),
    seed: int = 6,
    return_clean: bool = False,
):
    """Generate noisy cosines with distinct noise levels.

    Each signal is a cosine of the given ``period`` with a random phase offset and
    additive Gaussian noise. This is the synthetic dataset of the first example in
    the accompanying paper.

    Parameters
    ----------
    n_points : int, optional
        The number of time points.
    period : int, optional
        The period of the underlying cosines.
    noise_sds : sequence of float, optional
        The noise standard deviation for each signal; its length sets the number
        of signals.
    seed : int, optional
        A base seed; the signals use seeds ``seed``, ``seed + 1``, and so on.
    return_clean : bool, optional
        If ``True``, also return the noise-free signals (same phases) as a second
        DataFrame.

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        The noisy dataset, or ``(noisy, clean)`` if ``return_clean`` is ``True``.
    """
    t = np.linspace(0, n_points, n_points)
    noisy, clean = {}, {}
    for j, sd in enumerate(noise_sds):
        np.random.seed(seed + j)
        phase = np.random.uniform(0, 2 * period)
        signal = np.cos(2 * np.pi / period * (t - phase))
        noisy[j + 1] = signal + np.random.normal(0, sd, size=t.shape)
        clean[j + 1] = signal
    df = pd.DataFrame(noisy)
    return (df, pd.DataFrame(clean)) if return_clean else df


def generate_base_dataset(
    n_rows: int = 300,
    n_cols: int = 10,
    period: int = 50,
    seed: int = 42,
    permute: bool = True,
) -> pd.DataFrame:
    """Generate cosines, optionally scrambling all but the first three.

    This is the synthetic dataset of the second example in the accompanying paper,
    before noise is added (see :func:`add_noise`). The first three columns are cosines with random phase offsets; with
    ``permute=True`` the remaining columns are randomly permuted in time, removing
    their periodic structure.

    Parameters
    ----------
    n_rows : int, optional
        The number of time points.
    n_cols : int, optional
        The number of signals.
    period : int, optional
        The period of the underlying cosines.
    seed : int, optional
        The random seed.
    permute : bool, optional
        If ``True`` (default), time-scramble columns four onward. If ``False``,
        leave all columns as clean cosines (the phases are identical either way).

    Returns
    -------
    pandas.DataFrame
        The dataset, with string column labels ``"1"`` through ``"n_cols"``.
    """
    np.random.seed(seed)
    x = np.arange(n_rows)
    data = np.zeros((n_rows, n_cols))
    for col in range(n_cols):
        phase = np.random.uniform(0, 2 * np.pi)
        data[:, col] = np.cos(2 * np.pi * x / period + phase)
    if permute:
        for col in range(3, n_cols):
            np.random.shuffle(data[:, col])
    return pd.DataFrame(data, columns=[str(i + 1) for i in range(n_cols)])


def add_noise(
    df: pd.DataFrame,
    noise_sd: float = 0.1,
    master_seed: Optional[int] = None,
    instance_id: int = 0,
) -> pd.DataFrame:
    """Add i.i.d. Gaussian noise to every entry of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data.
    noise_sd : float, optional
        The noise standard deviation.
    master_seed : int, optional
        If given, ``master_seed`` and ``instance_id`` together select a
        reproducible seed, so distinct ``instance_id`` values give independent
        noise realisations. If ``None``, the noise is non-deterministic.
    instance_id : int, optional
        The realisation index used with ``master_seed``.

    Returns
    -------
    pandas.DataFrame
        A noisy copy of ``df``.
    """
    if master_seed is not None:
        rng = np.random.default_rng(master_seed)
        seeds = rng.integers(0, 1_000_000, size=instance_id + 1)
        np.random.seed(int(seeds[instance_id]))
    else:
        np.random.seed()
    return df + np.random.normal(0, noise_sd, size=df.shape)