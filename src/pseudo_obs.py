"""Leave-one-out KM pseudo-observations at a landmark time t0."""
import numpy as np
from lifelines import KaplanMeierFitter


def compute_pseudo_obs(T, Delta, t0=None):
    """Compute pseudo-observations θ_i = n·Ŝ(t0) − (n−1)·Ŝ_{−i}(t0).

    T: observed times, Delta: event indicators.
    t0: landmark time (default: median event time).
    Returns array of n pseudo-observations in [0, 1].
    """
    T     = np.asarray(T,     dtype=float)
    Delta = np.asarray(Delta, dtype=float)
    n     = len(T)

    if t0 is None:
        t0 = float(np.median(T[Delta == 1]))

    def km_surv(t_arr, d_arr, at):
        kmf = KaplanMeierFitter()
        kmf.fit(t_arr, event_observed=d_arr, label="")
        return float(kmf.survival_function_at_times(at).iloc[0])

    s_full = km_surv(T, Delta, t0)

    theta = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        s_loo   = km_surv(T[mask], Delta[mask], t0)
        theta[i] = n * s_full - (n - 1) * s_loo

    return theta
