"""Persistence-based objectives and projected stochastic subgradient descent.

The central routine is :func:`stochastic_subgradient_descent`, which optimizes a
function of the persistence diagram of a convex combination of distance matrices
over the probability simplex.
"""

from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from ripser import ripser

from .data_utils import (
    extract_vectors,
    get_matrix_indices,
    weighted_matrix_sum,
)


def persistence(mat: np.ndarray, dim: int = 1) -> np.ndarray:
    """Compute a Vietoris-Rips persistence diagram from a distance matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        A (possibly upper-triangular) distance matrix.
    dim : int, optional
        The homological degree.

    Returns
    -------
    numpy.ndarray
        An array of ``(birth, death)`` pairs, sorted by decreasing lifetime.
    """
    diagram = ripser(mat, distance_matrix=True, maxdim=dim)["dgms"][dim]
    if len(diagram) == 0:
        return diagram
    return np.stack(sorted(diagram, key=lambda bd: bd[1] - bd[0], reverse=True))


def _index_map(values, indices):
    table = defaultdict(list)
    for value, index in zip(values, indices):
        table[value].append(tuple(index))
    return table


def birth_death_indices(
    mat: np.ndarray,
    dim: int = 1,
    num_of_pairs: Union[int, str] = "all",
):
    """Locate the birth and death simplices of the top persistence pairs.

    Parameters
    ----------
    mat : numpy.ndarray
        A distance matrix.
    dim : int, optional
        The homological degree.
    num_of_pairs : int or "all", optional
        The number of pairs to return, ordered by decreasing lifetime.

    Returns
    -------
    tuple of list
        ``(birth_indices, death_indices)``, each a list of ``(row, col)`` pairs.
    """
    diagram = persistence(mat, dim)
    if num_of_pairs == "all":
        num_of_pairs = len(diagram)
    diagram = diagram[:num_of_pairs]

    births, deaths = diagram[:, 0], diagram[:, 1]
    birth_idx = get_matrix_indices(mat, births)
    death_idx = get_matrix_indices(mat, deaths)
    if num_of_pairs == 1:
        return birth_idx, death_idx

    birth_map = _index_map(mat[tuple(birth_idx.T)], birth_idx)
    death_map = _index_map(mat[tuple(death_idx.T)], death_idx)
    birth_sorted = [birth_map[b].pop(0) for b in births]
    death_sorted = [death_map[d].pop(0) for d in deaths]
    return birth_sorted, death_sorted


def persistence_gradient(
    bd_indices,
    matrices: np.ndarray,
    birth_weights: Optional[np.ndarray] = None,
    death_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a subgradient of the persistence objective in the weights.

    For each bar the contribution is ``death_weight * D[death] - birth_weight *
    D[birth]`` summed over the component matrices; the defaults correspond to
    (total or maximum) persistence.

    Parameters
    ----------
    bd_indices : tuple of list
        ``(birth_indices, death_indices)`` from :func:`birth_death_indices`.
    matrices : numpy.ndarray
        The component distance matrices, of shape ``(p, m, m)``.
    birth_weights, death_weights : numpy.ndarray, optional
        Per-bar weights; default to ``-1`` on births and ``+1`` on deaths.

    Returns
    -------
    numpy.ndarray
        The gradient vector, of shape ``(p,)``.
    """
    n_bars = len(bd_indices[0])
    birth_vecs = extract_vectors(matrices, bd_indices[0])
    death_vecs = extract_vectors(matrices, bd_indices[1])
    if birth_weights is None:
        birth_weights = -np.ones(n_bars)
    if death_weights is None:
        death_weights = np.ones(n_bars)
    return np.sum(
        birth_weights[:, None] * birth_vecs + death_weights[:, None] * death_vecs,
        axis=0,
    )


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project ``v`` onto the probability simplex ``{x >= 0, sum(x) = 1}``.

    Implements the algorithm of Duchi et al. (2008).

    Parameters
    ----------
    v : numpy.ndarray
        The point to project.

    Returns
    -------
    numpy.ndarray
        The nearest point of the simplex.
    """
    if np.all(v >= 0) and np.isclose(np.sum(v), 1.0):
        return v
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u > cssv / (np.arange(n) + 1))[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def estimate_grad_norm(
    matrices: Union[list, np.ndarray],
    num_of_features: Union[int, str] = 1,
) -> float:
    """Return the projected-subgradient norm at the barycenter.

    This is a convenient scale for the optimizer's hyperparameters: setting
    ``eta_0 = step / norm`` makes the initial step length equal to ``step``, and
    setting ``sigma = fraction * norm`` makes the noise a fixed fraction of the
    initial gradient.

    Parameters
    ----------
    matrices : numpy.ndarray
        The component distance matrices, of shape ``(p, m, m)``.
    num_of_features : int or "all", optional
        The number of persistence pairs in the objective.

    Returns
    -------
    float
        The Euclidean norm of the projected subgradient at the barycenter.
    """
    matrices = np.asarray(matrices)
    p = len(matrices)
    v = np.full(p, 1.0 / p)
    bd = birth_death_indices(weighted_matrix_sum(matrices, v), num_of_pairs=num_of_features)
    grad = persistence_gradient(bd, matrices)
    proj = (-grad) - np.mean(-grad)
    return float(np.linalg.norm(proj))


def stochastic_subgradient_descent(
    matrices: Union[list, np.ndarray],
    num_of_steps: int,
    num_of_features: Union[int, str] = 1,
    eta_0: float = 0.1,
    d: float = 0.01,
    sigma: float = 0.1,
    col_names: Optional[List] = None,
) -> pd.DataFrame:
    """Maximize a persistence functional over the simplex.

    Starting from the barycenter, the iteration is

    ``v_{i+1} = proj(v_i - eta_i (y_i + zeta_i))``,

    where ``eta_i = eta_0 / (1 + d i)`` is the learning rate, ``y_i`` is the
    projected subgradient of the negated objective, ``zeta_i ~ N(0, sigma^2 I)``
    is isotropic noise on the simplex's tangent space, and ``proj`` is the
    Euclidean projection onto the probability simplex.

    Parameters
    ----------
    matrices : numpy.ndarray
        The component distance matrices, of shape ``(p, m, m)``.
    num_of_steps : int
        The maximum number of iterations.
    num_of_features : int or "all", optional
        The number of persistence pairs in the objective (``1`` for maximum
        persistence).
    eta_0 : float, optional
        The initial learning rate. The effective step length at iteration ``i`` is
        ``eta_0 / (1 + d i)`` times the subgradient norm; choosing
        ``eta_0 = step / estimate_grad_norm(matrices)`` fixes the initial step
        length to ``step``.
    d : float, optional
        The learning-rate decay factor. Choosing ``d = 1 / num_of_steps`` halves
        the step length over the run; ``d = 0`` keeps it constant.
    sigma : float, optional
        The standard deviation of the gradient noise, in the same units as the
        subgradient. Choosing ``sigma = fraction * estimate_grad_norm(matrices)``
        sets the noise to a fixed fraction of the initial gradient.
    col_names : list, optional
        Column labels for the returned path.

    Returns
    -------
    pandas.DataFrame
        The path ``v_0, v_1, ...`` with one row per iterate. The final row is the
        score vector. The path may be shorter than ``num_of_steps + 1`` if the
        iteration reaches a fixed point.
    """
    matrices = np.asarray(matrices)
    p = len(matrices)
    v = np.full(p, 1.0 / p)
    path = [v.copy()]
    tol = 1e-5

    for i in range(num_of_steps):
        eta = eta_0 / (1.0 + d * i)

        weighted = weighted_matrix_sum(matrices, v)
        bd = birth_death_indices(weighted, num_of_pairs=num_of_features)
        subgrad = -persistence_gradient(bd, matrices)
        subgrad -= np.mean(subgrad)

        if np.linalg.norm(subgrad) < tol:
            break

        if sigma > 0:
            noise = np.random.randn(p)
            noise = (noise - np.mean(noise)) * sigma
        else:
            noise = 0.0

        new_v = project_to_simplex(v - eta * (subgrad + noise))
        if np.allclose(new_v, v, atol=tol):
            break
        v = new_v
        path.append(v.copy())

    result = pd.DataFrame(path)
    if col_names is not None:
        result.columns = col_names
    return result


def iter_grad_descent(
    dataset: pd.DataFrame,
    window: int,
    num_of_iter: int,
    num_of_steps: int,
    num_of_features: Union[int, str] = 1,
    eta_0: float = 0.1,
    d: float = 0.01,
    sigma: float = 0.1,
    col_names: Optional[List] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
    progress_every: int = 1,
) -> List[pd.DataFrame]:
    """Run the descent repeatedly and collect the resulting paths.

    The optimizer is stochastic, so independent runs on the same data trace
    different gradient paths. Averaging these paths estimates the mean gradient
    path.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The input time series.
    window : int
        The sliding-window length.
    num_of_iter : int
        The number of independent runs.
    num_of_steps : int
        The number of iterations per run.
    num_of_features : int or "all", optional
        The number of persistence pairs in the objective.
    eta_0, d, sigma : float, optional
        Hyperparameters for :func:`stochastic_subgradient_descent`.
    col_names : list, optional
        Column labels for the returned paths.
    seed : int, optional
        If given, seeds the sequence of runs for reproducibility.
    verbose : bool, optional
        If True, print progress messages.
    progress_every : int, optional
        Print progress every ``progress_every`` runs.

    Returns
    -------
    list of pandas.DataFrame
        One gradient path per run.
    """
    from .data_utils import sw_dist_mats

    if seed is not None:
        np.random.seed(seed)

    names = col_names if col_names is not None else list(dataset.columns)

    if verbose:
        print(
            f"Computing sliding-window distance matrices "
            f"(window={window}, variables={dataset.shape[1]})...",
            flush=True,
        )

    mats = sw_dist_mats(dataset, window=window)

    if verbose:
        print(
            f"Starting {num_of_iter} stochastic gradient descent runs "
            f"({num_of_steps} max steps each)...",
            flush=True,
        )

    paths = []

    for run_idx in range(num_of_iter):
        path = stochastic_subgradient_descent(
            mats,
            num_of_steps=num_of_steps,
            num_of_features=num_of_features,
            eta_0=eta_0,
            d=d,
            sigma=sigma,
            col_names=names,
        )
        paths.append(path)

        if verbose and (
            (run_idx + 1) == 1
            or (run_idx + 1) % progress_every == 0
            or (run_idx + 1) == num_of_iter
        ):
            print(
                f"Completed run {run_idx + 1}/{num_of_iter} "
                f"(path length: {len(path)})",
                flush=True,
            )

    if verbose:
        print("Finished all gradient descent runs.", flush=True)

    return paths

def mean_gradient_path(paths: List[pd.DataFrame]) -> pd.DataFrame:
    """Average a collection of gradient paths pointwise.

    Paths that terminate early are padded by repeating their final iterate so
    that all paths share the longest length before averaging.

    Parameters
    ----------
    paths : list of pandas.DataFrame
        The gradient paths to average.

    Returns
    -------
    pandas.DataFrame
        The mean path.
    """
    max_len = max(len(p) for p in paths)
    padded = []
    for p in paths:
        if len(p) < max_len:
            tail = pd.concat([p.iloc[[-1]]] * (max_len - len(p)), ignore_index=True)
            p = pd.concat([p, tail], ignore_index=True)
        padded.append(p)
    stacked = np.stack([p.values for p in padded])
    return pd.DataFrame(np.mean(stacked, axis=0), columns=paths[0].columns)


def scores(paths: List[pd.DataFrame]) -> pd.DataFrame:
    """Collect the final score vector from each path.

    Parameters
    ----------
    paths : list of pandas.DataFrame
        The gradient paths.

    Returns
    -------
    pandas.DataFrame
        One row per path, holding its final iterate.
    """
    return pd.DataFrame([p.iloc[-1].values for p in paths], columns=paths[0].columns)
