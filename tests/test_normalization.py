"""Prueba 2 — Normalización: escala y signo estables entre reestimaciones consecutivas."""
import numpy as np

import config
from cointegration.johansen import normalize_vector
from cointegration.stability import unit_normalize
from kalman.pipeline import order_assets, run_filters


def test_normalization_fixes_scale_and_sign():
    """Un eigenvector solo está definido salvo escala: cualquier múltiplo debe normalizar igual."""
    vector = np.array([0.8, 0.15, -0.5, -0.08])
    for factor in (-3.0, -1.0, 0.5, 10.0):
        assert np.allclose(normalize_vector(factor * vector), normalize_vector(vector))
        assert np.allclose(unit_normalize(factor * vector), unit_normalize(vector))
    assert np.isclose(normalize_vector(vector)[0], 1.0)


def test_kalman_weights_keep_scale_and_sign(log_prices, n_train, vector):
    """Los pesos del filtro 1 conservan la pata normalizada en +1, el signo de cada
    peso y cambios diarios pequeños entre reestimaciones consecutivas."""
    _, beta0 = order_assets(vector)
    f1, _, _ = run_filters(log_prices, beta0, n_train, config.KALMAN)
    weights = f1[[f"beta_{a}" for a in log_prices.columns]]

    assert (weights.iloc[:, 0] == 1.0).all()
    assert (np.sign(weights.iloc[:, 1:]) == np.sign(beta0[1:])).all().all()
    assert weights.diff().abs().max().max() < 0.01
