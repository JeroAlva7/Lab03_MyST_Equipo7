"""Filtro de Kalman 2: media y varianza latentes del spread para un z-score causal (sección 3.4).

El spread revierte a su media como un AR(1): S_t - mu_t = phi (S_{t-1} - mu_{t-1}) + e_t.
Con mu_t ~ mu_{t-1} queda un modelo lineal en el estado mu_t:
    Estado:      mu_t = mu_{t-1} + w_t,                     w ~ N(0, Q)
    Observación: y_t = S_t - phi S_{t-1} = (1 - phi) mu_t + e_t,   e ~ N(0, R)
La varianza sigma_t^2 del spread se actualiza en paralelo con un EWMA de las
innovaciones al cuadrado. El z-score usa los estadísticos PRIOR: normalizar con la
media posterior usaría la observación del día para juzgarse a sí misma.
"""
import numpy as np
import pandas as pd


def fit_ar1(spread: pd.Series) -> dict:
    """Ajusta S_t = a + phi S_{t-1} + e_t por OLS (solo con datos de entrenamiento).

    Returns:
        phi, media de largo plazo a / (1 - phi), varianza de e y vida media -ln2 / ln(phi).
    """
    y, x = spread.to_numpy()[1:], spread.to_numpy()[:-1]
    design = np.column_stack([np.ones(len(x)), x])
    (a, phi), *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - a - phi * x
    return {"phi": float(phi), "mu": float(a / (1 - phi)),
            "resid_var": float(residuals.var()), "half_life": float(-np.log(2) / np.log(phi))}


def run_spread_filter(spread: pd.Series, phi: float, mu0: float, q: float, r: float,
                      p0: float, ewma_lambda: float) -> pd.DataFrame:
    """Corre el filtro 2 y calcula el z-score con estadísticos prior.

    Args:
        spread: spread del filtro 1.
        phi: coeficiente AR(1) del spread (entrenamiento).
        mu0: media inicial del spread (entrenamiento).
        q: varianza diaria del cambio de la media latente (Q).
        r: varianza de la innovación AR(1) (R) y valor inicial de sigma_e^2.
        p0: varianza inicial de la media latente.
        ewma_lambda: persistencia del EWMA de la varianza.

    Returns:
        Por fecha: media y desviación prior, z-score, innovación normalizada,
        ganancia de Kalman, varianza posterior y media posterior.
    """
    s = spread.to_numpy()
    c = 1.0 - phi
    mu, p_var, sigma_e2 = mu0, p0, r
    rows = [[np.nan] * 7]                      # la primera fecha no tiene S_{t-1}

    for t in range(1, len(s)):
        p_prior = p_var + q                                   # Predict
        sigma_spread = np.sqrt(sigma_e2 / (1 - phi ** 2))     # desviación del spread, prior
        z_score = (s[t] - mu) / sigma_spread                  # z con media y sigma PRIOR

        innovation = s[t] - phi * s[t - 1] - c * mu           # Update
        innovation_var = c * c * p_prior + r
        gain = p_prior * c / innovation_var
        innovation_norm = innovation / np.sqrt(sigma_e2)
        rows.append([mu, sigma_spread, z_score, innovation_norm, gain, 0.0, 0.0])

        mu = mu + gain * innovation
        p_var = (1 - gain * c) * p_prior
        sigma_e2 = ewma_lambda * sigma_e2 + (1 - ewma_lambda) * innovation ** 2
        rows[-1][5], rows[-1][6] = p_var, mu

    columns = ["mu_prior", "sigma_prior", "z", "innovation_norm", "gain", "var_post", "mu_post"]
    return pd.DataFrame(rows, index=spread.index, columns=columns)


def log_likelihood(spread: pd.Series, phi: float, mu0: float, q: float, r: float, p0: float) -> float:
    """Log-verosimilitud gaussiana de las innovaciones del filtro 2 (para elegir Q)."""
    s = spread.to_numpy()
    c, mu, p_var, total = 1.0 - phi, mu0, p0, 0.0
    for t in range(1, len(s)):
        p_prior = p_var + q
        innovation = s[t] - phi * s[t - 1] - c * mu
        innovation_var = c * c * p_prior + r
        total -= 0.5 * (np.log(2 * np.pi * innovation_var) + innovation ** 2 / innovation_var)
        gain = p_prior * c / innovation_var
        mu, p_var = mu + gain * innovation, (1 - gain * c) * p_prior
    return float(total)
