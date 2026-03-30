"""Frozen-sigma AFT: TabPFN backbone + JAX/Optax sigma fitting."""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.stats import norm
from tabpfn import TabPFNRegressor

EPS = 1e-8
TABPFN_MAX = 10_000


def predict_log_time(X_train, y_train, X_test):
    """In-context regression via TabPFN. y_train: log-times of uncensored subjects."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if len(X_train) > TABPFN_MAX:
        idx = np.random.choice(len(X_train), TABPFN_MAX, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
    model = TabPFNRegressor(device="cpu")
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_sigma(T, Delta, mu_log, n_steps=500, lr=0.01):
    """Fit σ by censored log-likelihood MLE via Optax.

    T: observed times, Delta: event indicators, mu_log: predicted log-times (all train subjects).
    """
    log_t = jnp.log(jnp.clip(jnp.array(T, dtype=float), EPS, None))
    mu = jnp.array(mu_log, dtype=float)
    d = jnp.array(Delta, dtype=float)

    def neg_ll(log_sigma):
        sigma = jnp.exp(log_sigma) + EPS
        z = (log_t - mu) / sigma
        log_f = -log_t - jnp.log(sigma) + jax.scipy.stats.norm.logpdf(z)
        log_S = jax.scipy.stats.norm.logcdf(-z)   # log(1 - Phi(z))
        return -jnp.sum(d * log_f + (1 - d) * log_S)

    log_sigma = jnp.zeros(())
    opt = optax.adam(lr)
    state = opt.init(log_sigma)
    grad_fn = jax.jit(jax.value_and_grad(neg_ll))

    for _ in range(n_steps):
        _, g = grad_fn(log_sigma)
        updates, state = opt.update(g, state)
        log_sigma = optax.apply_updates(log_sigma, updates)

    return float(jnp.exp(log_sigma))


def survival_lognormal(t_grid, mu_log, sigma):
    """S(t | x) = 1 - Φ((log t - μ) / σ).

    t_grid: (T,)  mu_log: (n,)  →  returns (n, T).
    """
    z = (np.log(np.clip(t_grid[None, :], EPS, None)) - np.asarray(mu_log)[:, None]) / sigma
    return norm.sf(z)


def predicted_median(surv_matrix, t_grid):
    """First t where S(t) ≤ 0.5; inf if never crosses within the grid."""
    t_grid = np.asarray(t_grid)
    crossed = surv_matrix <= 0.5
    has = crossed.any(axis=1)
    idx = crossed.argmax(axis=1)
    med = np.full(len(surv_matrix), np.inf)
    med[has] = t_grid[idx[has]]
    return med
