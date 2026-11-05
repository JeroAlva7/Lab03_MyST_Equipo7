"""Filtro de Kalman 1: pesos dinámicos del portafolio cointegrado (sección 3.3).

Modelo de espacio de estados (S13 y S14), con la pata normalizada (beta = +1) como
observación:
    Estado:      z_t = (c_t, beta_2,t, ..., beta_n,t),   z_t = A z_{t-1} + w_t,  w ~ N(0, Q),  A = I
    Observación: y_t = ln P1_t = C_t z_t + v_t,           C_t = (1, -ln P2_t, ..., -ln Pn_t),  v ~ N(0, R)

Así el spread es S_t = beta_t^T ln P_t = ln P1_t + sum(beta_i ln Pi_t) = c_t + v_t.
"""
import numpy as np
import pandas as pd


def run_weights_filter(log_prices: pd.DataFrame, beta0: np.ndarray, intercept0: float,
                       q: float, r: float, p0: float) -> pd.DataFrame:
    """Corre el filtro 1 sobre toda la muestra.

    Args:
        log_prices: log-precios; la primera columna es la pata normalizada (beta = +1).
        beta0: vector de Johansen normalizado (beta0[0] = 1), estado inicial de los pesos.
        intercept0: media del spread de Johansen en entrenamiento, estado inicial de c.
        q: varianza diaria del cambio de cada peso (Q = q I, "drift velocity").
        r: varianza del ruido de observación (R, "price noise").
        p0: varianza inicial del estado (P_0 = p0 I), confianza en el vector de Johansen.

    Returns:
        Por fecha: pesos e intercepto posteriores, spread calculado con los pesos prior
        (causal), innovación, su varianza, ganancia de Kalman y varianza posterior de
        cada componente del estado.
    """
    assets = list(log_prices.columns)
    states = ["intercept"] + assets[1:]
    x = log_prices.to_numpy()
    n_states = len(states)

    z = np.r_[intercept0, beta0[1:]]          # estado posterior z_{t-1|t-1}
    p_cov = np.eye(n_states) * p0              # covarianza posterior P_{t-1|t-1}
    q_cov = np.eye(n_states) * q
    rows = []

    for t in range(len(x)):
        # Fase 1 — Predict (A = I, caminata aleatoria)
        z_prior = z
        p_prior = p_cov + q_cov
        # Spread con los pesos prior: solo usa información hasta t-1 para los pesos
        spread = x[t] @ np.r_[1.0, z_prior[1:]]

        # Fase 2 — Update
        c_t = np.r_[1.0, -x[t, 1:]]
        innovation = x[t, 0] - c_t @ z_prior                  # y_t - C_t z_{t|t-1}
        innovation_var = c_t @ p_prior @ c_t + r              # S_t = C P C^T + R
        gain = p_prior @ c_t / innovation_var                 # K_t = P C^T S^-1
        z = z_prior + gain * innovation                       # nuevo = viejo + ganancia x sorpresa
        p_cov = (np.eye(n_states) - np.outer(gain, c_t)) @ p_prior

        rows.append([*z, spread, innovation, innovation_var, *gain, *np.diag(p_cov)])

    columns = (["intercept"] + [f"beta_{a}" for a in assets[1:]]
               + ["spread", "innovation", "innovation_var"]
               + [f"gain_{s}" for s in states] + [f"var_{s}" for s in states])
    out = pd.DataFrame(rows, index=log_prices.index, columns=columns)
    out.insert(1, f"beta_{assets[0]}", 1.0)
    return out
