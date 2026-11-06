"""Tamaño de la posición en cada pata (sección 3.1 y 3.6).

Al abrir o rebalancear, el nocional bruto total es el 80% del capital disponible y se reparte
entre los activos en proporción a |beta_i| del vector de pesos del filtro 1:

    nocional_i = direction * signo(beta_i) * f * capital * |beta_i| / sum_j |beta_j|
    acciones_i = nocional_i / precio_i

- direction = +1 (spread largo): se compran las patas con beta > 0 y se venden las de beta < 0.
- direction = -1 (spread corto): se invierte todo.
- La suma de |nocional_i| es exactamente f * capital; el 20% restante queda en efectivo.

Supuesto: se permiten fracciones de acción, para que el nocional respete exactamente la
proporción del vector (con $1,000,000 el redondeo a acciones enteras cambiaría los pesos
en menos de 0.01%).
"""
import numpy as np
import pandas as pd


def kalman_weights(filter1: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Vector de pesos beta_t del filtro 1, una columna por activo en el orden de `assets`."""
    return filter1[[f"beta_{a}" for a in assets]].set_axis(assets, axis=1)


def target_notional(direction: int, weights: np.ndarray, capital: float, fraction: float) -> np.ndarray:
    """Nocional objetivo en dólares de cada pata (positivo = largo, negativo = corto).

    Args:
        direction: +1 spread largo, -1 spread corto, 0 sin posición.
        weights: vector de pesos beta_t (uno por activo).
        capital: capital disponible (valor del portafolio) al momento de operar.
        fraction: fracción del capital que se despliega (config.CAPITAL_FRACTION = 0.80).
    """
    weights = np.asarray(weights, dtype=float)
    return direction * fraction * capital * weights / np.abs(weights).sum()


def target_shares(direction: int, weights: np.ndarray, prices: np.ndarray,
                  capital: float, fraction: float) -> np.ndarray:
    """Acciones objetivo de cada pata: nocional objetivo / precio."""
    return target_notional(direction, weights, capital, fraction) / np.asarray(prices, dtype=float)
