"""VECM con el vector de Johansen como término de corrección de error (sección 3.6).

Modelo, con X_t = log-precios y el vector beta fijo (estimado por Johansen en entrenamiento):

    dX_t = c + alpha * ect_{t-1} + Gamma_1 dX_{t-1} + ... + Gamma_k dX_{t-k} + e_t
    ect_t = beta^T X_t   (desviación del equilibrio = spread)

Como beta ya está fijo, cada ecuación del VECM es una regresión lineal y se estima por OLS
(todas las ecuaciones comparten regresores, así que se resuelven juntas).
- alpha (loadings): qué tan rápido y en qué dirección corrige cada activo cuando el spread
  se aleja del equilibrio. Para que haya reversión, beta^T alpha debe ser negativo.
- Los residuales e_t son la parte de cada movimiento que el VECM no explica.

La señal no se recalcula aquí: es la columna "z" del filtro 2 (kalman/spread_filter.py),
la desviación del equilibrio normalizada con la media y la desviación PRIOR, es decir,
con estadísticos que solo usan información hasta t-1.
"""
import numpy as np
import pandas as pd


def build_design(log_prices: pd.DataFrame, beta: np.ndarray, k_ar_diff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arma las variables del VECM.

    Returns:
        (Y, Z): Y tiene dX_t (una columna por activo); Z tiene la constante, ect_{t-1}
        y los rezagos dX_{t-i} para i = 1..k_ar_diff. Ambas sin filas con faltantes.
    """
    diffs = log_prices.diff()
    regressors = {"ect_lag": (log_prices @ beta).shift(1)}
    for lag in range(1, k_ar_diff + 1):
        for asset in log_prices.columns:
            regressors[f"d{asset}_lag{lag}"] = diffs[asset].shift(lag)
    design = pd.DataFrame(regressors)
    design.insert(0, "const", 1.0)
    data = pd.concat([diffs, design], axis=1).dropna()
    return data[log_prices.columns], data[design.columns]


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mínimos cuadrados ordinarios para varias ecuaciones con los mismos regresores.

    b = (X'X)^-1 X'Y; el error estándar de cada coeficiente es sqrt(s^2 * diag((X'X)^-1)),
    con s^2 = suma de residuales al cuadrado / (n - k) de cada ecuación.

    Returns:
        (coeficientes k x m, residuales n x m, errores estándar k x m).
    """
    xtx_inv = np.linalg.inv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    residuals = y - x @ coef
    s2 = (residuals ** 2).sum(axis=0) / (len(y) - x.shape[1])
    std_err = np.sqrt(np.outer(np.diag(xtx_inv), s2))
    return coef, residuals, std_err


def fit_vecm(log_train: pd.DataFrame, beta: np.ndarray, k_ar_diff: int) -> dict:
    """Estima el VECM por OLS, una ecuación por activo, solo con datos de entrenamiento.

    Args:
        log_train: log-precios de entrenamiento (mismo orden de columnas que beta).
        beta: vector de Johansen normalizado.
        k_ar_diff: rezagos en diferencias (los mismos que usó Johansen).

    Returns:
        alpha y su t-estadístico por activo, matriz Gamma, constantes, covarianza de
        residuales, velocidad de corrección del spread (beta^T alpha) y su vida media.
    """
    y, z = build_design(log_train, beta, k_ar_diff)
    coef, residuals, std_err = ols(y.to_numpy(), z.to_numpy())
    coef = pd.DataFrame(coef, index=z.columns, columns=y.columns)
    std_err = pd.DataFrame(std_err, index=z.columns, columns=y.columns)

    alpha = coef.loc["ect_lag"]
    alpha_t = alpha / std_err.loc["ect_lag"]
    gamma = coef.drop(["const", "ect_lag"]).T
    residuals = pd.DataFrame(residuals, index=y.index, columns=y.columns)

    # Al multiplicar el VECM por beta^T: dS_t ~ (beta^T alpha) S_{t-1} + ...
    speed = float(beta @ alpha.to_numpy())
    half_life = float(-np.log(2) / np.log(1 + speed)) if -1 < speed < 0 else float("inf")

    return {"assets": list(y.columns), "k_ar_diff": k_ar_diff,
            "alpha": alpha.to_dict(), "alpha_t": alpha_t.to_dict(),
            "gamma": gamma.to_dict(orient="index"),
            "const": coef.loc["const"].to_dict(),
            "resid_cov": residuals.cov().to_dict(),
            "speed": speed, "half_life": half_life, "n_obs": int(len(y))}