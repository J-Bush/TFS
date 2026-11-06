"""Tests for the persistence pairs, the subgradient and the three-sine example."""

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import pairwise_distances

from src.data_utils import (
    generate_base_dataset,
    generate_three_sine_dataset,
    sliding_window,
    sw_dist_mats,
    weighted_matrix_sum,
)
from src.persistence_optimization import (
    birth_death_indices,
    estimate_grad_norm,
    persistence,
    persistence_gradient,
    stochastic_subgradient_descent,
)


def _rank_matrix(mat):
    """Ranks of the edge lengths, with ties broken by position in the matrix."""
    m = len(mat)
    rows, cols = np.triu_indices(m, k=1)
    order = np.argsort(mat[rows, cols], kind="stable")
    ranks = np.zeros((m, m))
    ranks[rows[order], cols[order]] = np.arange(1, len(order) + 1)
    return ranks + ranks.T


def _brute_force_h1_ranks(mat):
    """Degree-1 pairs of the Rips filtration of the rank matrix, by Z/2 column reduction."""
    m = len(mat)
    ranks = _rank_matrix(mat)
    simplices = [(0.0, (i,)) for i in range(m)]
    simplices += [(ranks[i, j], (i, j)) for i, j in itertools.combinations(range(m), 2)]
    simplices += [
        (max(ranks[i, j], ranks[i, k], ranks[j, k]), (i, j, k))
        for i, j, k in itertools.combinations(range(m), 3)
    ]
    simplices.sort(key=lambda s: (s[0], len(s[1]), s[1]))
    position = {s: n for n, (_, s) in enumerate(simplices)}
    columns = [
        {position[f] for f in itertools.combinations(s, len(s) - 1)} if len(s) > 1 else set()
        for _, s in simplices
    ]
    pivot_of, pairs = {}, []
    for j, col in enumerate(columns):
        while col and max(col) in pivot_of:
            col ^= columns[pivot_of[max(col)]]
        if col:
            i = max(col)
            pivot_of[i] = j
            if len(simplices[i][1]) == 2 and simplices[j][0] > simplices[i][0]:
                pairs.append((int(simplices[i][0]), int(simplices[j][0])))
    return sorted(pairs)


def _rank_pairs(mat):
    ranks = _rank_matrix(mat)
    births, deaths = birth_death_indices(mat, num_of_pairs="all")
    return sorted((int(ranks[i, j]), int(ranks[k, l])) for (i, j), (k, l) in zip(births, deaths))


def _circles(rng, n=20, k=2, noise=0.15):
    theta = rng.uniform(0, 2 * np.pi, n)
    offset = rng.integers(0, k, n) * 3.0
    pts = np.column_stack([np.cos(theta) + offset, np.sin(theta)])
    return pts + rng.normal(scale=noise, size=pts.shape)


@pytest.mark.parametrize("seed", range(4))
def test_pairs_match_brute_force(seed):
    rng = np.random.default_rng(seed)
    mat = np.triu(pairwise_distances(_circles(rng)), k=1)
    assert _rank_pairs(mat) == _brute_force_h1_ranks(mat)


@pytest.mark.parametrize("seed", range(3))
def test_pairs_with_many_tied_lengths(seed):
    rng = np.random.default_rng(seed)
    pts = np.round(_circles(rng, n=18) * 2)
    mat = np.triu(pairwise_distances(pts, metric="manhattan"), k=1)
    assert len(np.unique(mat)) < 30  # many equal lengths
    assert _rank_pairs(mat) == _brute_force_h1_ranks(mat)


@pytest.mark.parametrize("seed", range(4))
def test_lifetimes_match_ripser(seed):
    rng = np.random.default_rng(seed)
    mat = np.triu(pairwise_distances(_circles(rng, n=40, k=3)), k=1)
    births, deaths = birth_death_indices(mat)
    lifetimes = mat[tuple(deaths.T)] - mat[tuple(births.T)]
    assert np.all(np.diff(lifetimes) <= 0)  # sorted by decreasing lifetime
    reference = persistence(mat)
    reference = reference[:, 1] - reference[:, 0]
    np.testing.assert_allclose(lifetimes[lifetimes > 0], reference, rtol=1e-5, atol=1e-5)


def test_symmetric_and_upper_triangular_inputs_agree():
    rng = np.random.default_rng(0)
    upper = np.triu(pairwise_distances(_circles(rng, n=30)), k=1)
    for a, b in zip(birth_death_indices(upper), birth_death_indices(upper + upper.T)):
        np.testing.assert_array_equal(a, b)


def _example_matrices():
    data = generate_three_sine_dataset(n_points=300, period=50, noise_sds=(1, 2, 3), seed=6)
    return sw_dist_mats(data, window=250)


@pytest.mark.parametrize("num_of_pairs", [1, 3, "all"])
def test_gradient_matches_finite_differences(num_of_pairs):
    mats = _example_matrices()
    rng = np.random.default_rng(1)

    def objective(v):
        mat = weighted_matrix_sum(mats, v)
        births, deaths = birth_death_indices(mat, num_of_pairs=num_of_pairs)
        return np.sum(mat[tuple(deaths.T)] - mat[tuple(births.T)])

    for _ in range(4):
        v = rng.dirichlet(np.ones(3))
        u = rng.normal(size=3)
        u -= u.mean()
        h = 1e-7
        finite_diff = (objective(v + h * u) - objective(v - h * u)) / (2 * h)
        grad = persistence_gradient(
            birth_death_indices(weighted_matrix_sum(mats, v), num_of_pairs=num_of_pairs), mats
        )
        assert finite_diff == pytest.approx(grad @ u, rel=1e-5, abs=1e-5)


def test_worked_example_selects_least_noisy_signal():
    mats = _example_matrices()
    norm = estimate_grad_norm(mats, num_of_features=1)
    assert norm > 0
    path = stochastic_subgradient_descent(
        mats,
        num_of_steps=500,
        num_of_features=1,
        eta_0=(2 ** -4) / 3 / norm,
        d=1 / 500,
        sigma=0.1 * norm,
        col_names=["1", "2", "3"],
        seed=0,
    )
    assert path.iloc[-1]["1"] > 0.99
    rerun = stochastic_subgradient_descent(
        mats, 500, 1, (2 ** -4) / 3 / norm, 1 / 500, 0.1 * norm, ["1", "2", "3"], seed=0
    )
    pd.testing.assert_frame_equal(path, rerun)


def test_sliding_window_layout():
    data = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    expected = [[0, 10, 1, 11, 2, 12], [1, 11, 2, 12, 3, 13]]
    np.testing.assert_array_equal(sliding_window(data, 3).values, expected)
    np.testing.assert_array_equal(sliding_window([5, 6, 7], 2).values, [[5, 6], [6, 7]])
    with pytest.raises(ValueError):
        sliding_window(data, 5)


def test_l1_global_matrix_is_sum_of_components():
    data = generate_base_dataset(n_rows=80, n_cols=4)
    np.testing.assert_allclose(
        sw_dist_mats(data, window=20, glob=True), sw_dist_mats(data, window=20).sum(axis=0)
    )

