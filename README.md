# Topological feature selection for time series data

Tools from applied topology for scoring the variables of a multivariate time
series by their contribution to the topology of its sliding-window embedding.

Given a multivariate time series, the method forms the sliding-window embedding
of each variable, summarizes it with a persistence diagram, and searches over
convex combinations of the variables for the combination that maximizes a chosen
functional of the diagram (for example, the lifetime of the most persistent
loop). The search uses projected stochastic subgradient descent on the
probability simplex. The resulting weights are interpreted as scores: variables
with larger weights contribute more to the coordinated, periodic structure of the
data.

This repository accompanies the paper *Topological Feature Selection for Time
Series Data* by Peter Bubenik and Johnathan Bush.


## Quick start

```python
import numpy as np
from src.data_utils import generate_three_sine_dataset, sw_dist_mats
from src.persistence_optimization import estimate_grad_norm, stochastic_subgradient_descent

# Three cosines with increasing noise.
dataset = generate_three_sine_dataset(noise_sds=(1, 2, 3), seed=6)

# One sliding-window distance matrix per signal.
dist_mats = sw_dist_mats(dataset, window=250)

# Maximize the maximum persistence over convex combinations of the signals.
step = (2 ** -4) / 3
norm = estimate_grad_norm(dist_mats, num_of_features=1)
path = stochastic_subgradient_descent(
    dist_mats,
    num_of_steps=500,
    num_of_features=1,
    eta_0=step / norm,
    d=1 / 500,
    sigma=0.1 * norm,
    col_names=["1", "2", "3"],
)

print(path.iloc[-1])   # scores
```

An annotated walkthrough is in
[`notebooks/three_sine_example.ipynb`](notebooks/three_sine_example.ipynb).

## Repository layout

```
src/
  data_utils.py               sliding windows, distance matrices, datasets
  persistence_optimization.py persistence objectives and Algorithm 6.2
  plotting_utils.py           figures (time series, diagrams, paths, scores, covariance)
notebooks/
  three_sine_example.ipynb    worked example
requirements.txt
LICENSE
```

## Method at a glance

For a chosen window length, the sliding-window embedding of each variable yields a
distance matrix `D_j`. For weights `v` on the simplex, the convex combination
`sum_j v_j D_j` defines a filtration whose degree-`k` persistence diagram is
denoted `PD(v)`. Given a functional `F` on diagrams, the method solves

```
maximize   F(PD(v))   subject to   v in the probability simplex,
```

by projected stochastic subgradient descent. Starting from the barycenter, each
iteration takes a noisy subgradient step and projects back onto the simplex:

```
v <- proj( v - eta * (subgradient + noise) ),
```

with a decaying learning rate. The hyperparameters are expressed
through `estimate_grad_norm`, which returns the subgradient norm at the
barycenter: setting `eta_0 = step / norm` fixes the initial step length, and
`sigma = fraction * norm` sets the noise to a fixed fraction of the initial
gradient.

## Citation

If you use this code, please cite the accompanying paper (https://arxiv.org/abs/2310.17494). A BibTeX entry will be added here on publication.

## License

Released under the MIT License. See [LICENSE](LICENSE).