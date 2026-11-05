"""Ejemplo numérico trabajado de ambos filtros (sección 3.5).

Reconstruye, para varios días consecutivos, cada paso de las ecuaciones de Kalman a partir
de la salida de los filtros, y verifica que la actualización cuadre:
nuevo estado = estado anterior + ganancia x sorpresa.
"""
import numpy as np
import pandas as pd


def worked_example_filter1(log_prices: pd.DataFrame, filter1: pd.DataFrame, q: float,
                           dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Paso a paso del filtro 1 para cada fecha de `dates`.

    Por activo libre i: peso prior (posterior del día anterior, A = I), ganancia K_i y peso
    posterior. Además: observación y_t = ln P1, predicción C_t z_prior, innovación, su
    varianza S_t y el error de reconstrucción |posterior - (prior + K * innovación)|.
    """
    assets = list(log_prices.columns)
    states = ["intercept"] + [f"beta_{a}" for a in assets[1:]]
    gains = ["gain_intercept"] + [f"gain_{a}" for a in assets[1:]]
    rows = []
    for date in dates:
        t = filter1.index.get_loc(date)
        prior = filter1[states].iloc[t - 1].to_numpy()
        post = filter1[states].iloc[t].to_numpy()
        gain = filter1[gains].iloc[t].to_numpy()
        c_t = np.r_[1.0, -log_prices.iloc[t, 1:].to_numpy()]
        innovation = filter1["innovation"].iloc[t]
        row = {"fecha": date.date(), "y_t (ln P1)": log_prices.iloc[t, 0],
               "prediccion C_t z_prior": c_t @ prior, "innovacion": innovation,
               "S_t": filter1["innovation_var"].iloc[t],
               "P_prior intercepto": filter1["var_intercept"].iloc[t - 1] + q}
        for name, p, k, z in zip(states, prior, gain, post):
            row.update({f"{name} prior": p, f"K {name}": k, f"{name} posterior": z})
        row["error de reconstruccion"] = float(np.max(np.abs(post - (prior + gain * innovation))))
        row["spread S_t"] = filter1["spread"].iloc[t]
        rows.append(row)
    return pd.DataFrame(rows).set_index("fecha")


def worked_example_filter2(spread: pd.Series, filter2: pd.DataFrame, params: dict,
                           dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Paso a paso del filtro 2: predict, z-score prior, innovación, ganancia y update de mu."""
    phi, q, r = params["phi"], params["q"], params["resid_var"]
    rows = []
    for date in dates:
        t = filter2.index.get_loc(date)
        mu_prior = filter2["mu_prior"].iloc[t]
        p_prior = filter2["var_post"].iloc[t - 1] + q
        y_t = spread.iloc[t] - phi * spread.iloc[t - 1]
        innovation = y_t - (1 - phi) * mu_prior
        innovation_var = (1 - phi) ** 2 * p_prior + r
        gain = filter2["gain"].iloc[t]
        mu_post = filter2["mu_post"].iloc[t]
        rows.append({"fecha": date.date(), "S_t": spread.iloc[t], "S_t-1": spread.iloc[t - 1],
                     "mu prior": mu_prior, "P prior": p_prior,
                     "sigma prior": filter2["sigma_prior"].iloc[t], "z_t": filter2["z"].iloc[t],
                     "y_t = S_t - phi S_t-1": y_t, "innovacion": innovation,
                     "varianza de la innovacion": innovation_var,
                     "K = P(1-phi)/var": p_prior * (1 - phi) / innovation_var, "K del filtro": gain,
                     "mu posterior": mu_post,
                     "error de reconstruccion": abs(mu_post - (mu_prior + gain * innovation))})
    return pd.DataFrame(rows).set_index("fecha")
