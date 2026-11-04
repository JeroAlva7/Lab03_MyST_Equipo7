"""Prueba de Johansen (paso 4 de la sección 3.2).

VECM: dP_t = Pi P_{t-1} + sum(Gamma_i dP_{t-i}) + e_t, con Pi = alpha beta^T.
El rango de Pi es el número de relaciones de cointegración. Se prueba
H0: rango <= r0 para r0 = 0, 1, ...; el primer r0 que no se rechaza es el rango.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

CRITICAL_COLUMN_95 = 1  # coint_johansen entrega valores críticos al 90%, 95% y 99%


def select_k_ar_diff(log_prices: pd.DataFrame, max_lags: int) -> int:
    """Elige los rezagos del VECM con el criterio BIC de un VAR en niveles.

    Un VAR(p) en niveles equivale a un VECM con p-1 rezagos en diferencias
    (derivación vista en clase). Se usa al menos 1 rezago.
    """
    p = VAR(log_prices.values).select_order(maxlags=max_lags).selected_orders["bic"]
    return max(int(p) - 1, 1)


def johansen_test(log_prices: pd.DataFrame, k_ar_diff: int) -> dict:
    """Corre Johansen con constante (det_order=0) y determina el rango.

    Returns:
        Eigenvalores, eigenvectores (columnas, de mayor a menor eigenvalor),
        estadísticos de traza y máximo eigenvalor con sus valores críticos al 95%,
        y el rango según cada estadístico.
    """
    # statsmodels resuelve un problema de eigenvalores no simétrico: puede devolver
    # partes imaginarias del orden de 1e-17 (error numérico). Se descartan de forma
    # explícita y se verifica que de verdad sean despreciables.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
        result = coint_johansen(log_prices.values, det_order=0, k_ar_diff=k_ar_diff)
    if np.max(np.abs(np.imag(result.evec))) > 1e-8:
        raise ValueError("Eigenvectores con parte imaginaria no despreciable")
    trace, trace_cv = result.lr1, result.cvt[:, CRITICAL_COLUMN_95]
    max_eig, max_eig_cv = result.lr2, result.cvm[:, CRITICAL_COLUMN_95]
    return {
        "eigenvalues": np.real(result.eig),
        "eigenvectors": np.real(result.evec),
        "trace": trace, "trace_cv95": trace_cv,
        "max_eig": max_eig, "max_eig_cv95": max_eig_cv,
        "rank_trace": first_non_rejection(trace, trace_cv),
        "rank_max_eig": first_non_rejection(max_eig, max_eig_cv),
        "k_ar_diff": k_ar_diff,
    }


def first_non_rejection(statistics: np.ndarray, critical_values: np.ndarray) -> int:
    """Rango estimado: primer r0 en que el estadístico no supera su valor crítico."""
    for r0, (stat, cv) in enumerate(zip(statistics, critical_values)):
        if stat < cv:
            return r0
    return len(statistics)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza el vector dividiendo entre la pata de mayor valor absoluto.

    Convención: esa pata queda con peso +1 y fija la escala y el signo del spread
    S_t = beta^T log P_t (sección 3.3 de los lineamientos).
    """
    return vector / vector[np.argmax(np.abs(vector))]
