"""Prueba 1 — Causalidad: el z-score en t no cambia al recalcular todo con datos hasta t."""
import numpy as np
import pytest

import config
from cointegration.johansen import johansen_test, normalize_vector
from kalman.pipeline import run_filters


def z_score_until(log_prices, n_train, k_ar_diff, last):
    """Recalcula el pipeline completo (Johansen + filtro 1 + filtro 2) con datos hasta `last`."""
    data = log_prices.iloc[:last + 1]
    beta0 = normalize_vector(johansen_test(data.iloc[:n_train], k_ar_diff)["eigenvectors"][:, 0])
    _, f2, _ = run_filters(data, beta0, n_train, config.KALMAN)
    return f2["z"]


@pytest.mark.parametrize("offset", [10, 300, 900])
def test_z_score_is_causal(log_prices, n_train, vector, offset):
    """Fechas en prueba (n_train + 10, + 300) y en validación (+ 900)."""
    t = n_train + offset
    z_full = z_score_until(log_prices, n_train, vector["k_ar_diff"], len(log_prices) - 1)
    z_truncated = z_score_until(log_prices, n_train, vector["k_ar_diff"], t)
    assert np.isclose(z_full.iloc[t], z_truncated.iloc[t], rtol=0, atol=1e-12)
    assert z_truncated.index[-1] == log_prices.index[t]
