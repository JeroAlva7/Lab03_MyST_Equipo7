"""Orquesta ambos filtros con parámetros estimados solo en entrenamiento.

Se separa de main.py para que las pruebas puedan recalcular el pipeline sobre datos
truncados (prueba de causalidad) sin escribir archivos.
"""
import numpy as np
import pandas as pd

from kalman.spread_filter import fit_ar1, log_likelihood, run_spread_filter
from kalman.weights_filter import run_weights_filter


def order_assets(vector: dict) -> tuple[list[str], np.ndarray]:
    """Pone primero la pata normalizada (beta = +1), que es la observación del filtro 1."""
    assets = sorted(vector["assets"], key=lambda a: abs(vector["beta"][a] - 1.0) > 1e-9)
    return assets, np.array([vector["beta"][a] for a in assets])


def run_filters(log_prices: pd.DataFrame, beta0: np.ndarray, n_train: int,
                settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Corre el filtro 1 y el filtro 2; todos los parámetros salen de las primeras `n_train` fechas.

    Returns:
        (salida del filtro 1, salida del filtro 2, parámetros usados).
    """
    # Filtro 1: R = varianza del spread de Johansen en entrenamiento ("price noise")
    johansen_spread = log_prices.iloc[:n_train] @ beta0
    r_weights = float(johansen_spread.var())
    f1 = run_weights_filter(log_prices, beta0, float(johansen_spread.mean()),
                            settings["q_weights"], r_weights, settings["p0_weights"])

    # Filtro 2: AR(1) del spread y Q por máxima verosimilitud, ambos en entrenamiento
    train_spread = f1["spread"].iloc[settings["burn_in"]:n_train]
    ar = fit_ar1(train_spread)
    likelihoods = {q: log_likelihood(train_spread, ar["phi"], ar["mu"], q, ar["resid_var"],
                                     settings["p0_mean"]) for q in settings["q_mean_grid"]}
    q_mean = max(likelihoods, key=likelihoods.get)
    f2 = run_spread_filter(f1["spread"], ar["phi"], ar["mu"], q_mean, ar["resid_var"],
                           settings["p0_mean"], settings["ewma_lambda"])

    params = {"filter1": {"q": settings["q_weights"], "r": r_weights, "p0": settings["p0_weights"]},
              "filter2": {**ar, "q": q_mean, "p0": settings["p0_mean"],
                          "ewma_lambda": settings["ewma_lambda"],
                          "log_likelihood_grid": {str(q): v for q, v in likelihoods.items()}}}
    return f1, f2, params
