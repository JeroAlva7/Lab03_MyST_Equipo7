"""Prueba 4 — Filtro: innovaciones normalizadas con varianza cercana a 1 y sin autocorrelación.

Tolerancia declarada: varianza entre 0.8 y 1.2 y Ljung-Box (10 rezagos) con p > 0.05,
evaluadas fuera de muestra (prueba y validación). En entrenamiento la prueba falla por la
volatilidad de marzo de 2020; se documenta en el reporte.
"""
import pytest

import config
from data.validation import split_data
from kalman.diagnostics import innovation_diagnostics
from kalman.pipeline import order_assets, run_filters


@pytest.mark.parametrize("split", ["test", "val"])
def test_innovations_are_white_noise(prices, log_prices, n_train, vector, split):
    """Innovaciones normalizadas del filtro 2 fuera de muestra: varianza cercana a 1 y sin autocorrelación."""
    _, beta0 = order_assets(vector)
    _, f2, _ = run_filters(log_prices, beta0, n_train, config.KALMAN)
    dates = split_data(prices, config.SPLIT)[split].index
    d = innovation_diagnostics(f2.loc[dates, "innovation_norm"])
    assert 0.8 < d["variance"] < 1.2
    assert d["ljung_box_p"] > 0.05
