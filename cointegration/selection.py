"""Embudo de selección del grupo (sección 3.2) y validación del spread (paso 5).

Etapas en orden: correlación -> Engle-Granger -> Johansen -> validación del
spread (ADF y vida media). Todo se calcula solo con la partición de entrenamiento.
"""
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from cointegration.correlation import group_correlation, rolling_pair_correlation
from cointegration.engle_granger import engle_granger_group
from cointegration.integration import adf_pvalue
from cointegration.johansen import johansen_test, normalize_vector, select_k_ar_diff


def half_life(spread: pd.Series) -> float:
    """Vida media de la reversión: dS_t = lambda (S_{t-1} - media) + e_t, HL = -ln2 / lambda.

    Regresa infinito si lambda >= 0 (el spread no revierte).
    """
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lam = sm.OLS(delta, sm.add_constant(lagged)).fit().params.iloc[1]
    return float(-np.log(2) / lam) if lam < 0 else float("inf")


def candidate_groups(assets: list[str]) -> list[tuple[str, ...]]:
    """Todas las combinaciones de 2 o más activos del universo (11 para 4 activos)."""
    return [g for size in range(2, len(assets) + 1) for g in combinations(assets, size)]


def evaluate_vectors(log_prices: pd.DataFrame, johansen: dict, rank: int) -> list[dict]:
    """Construye el spread de cada vector cointegrante y mide ADF y vida media.

    Cuando el rango es mayor que 1 se evalúan todas las columnas para seleccionar
    empíricamente (nota de la sección 3.2).
    """
    results = []
    for column in range(rank):
        beta = normalize_vector(johansen["eigenvectors"][:, column])
        spread = log_prices @ beta
        results.append({"column": column, "beta": beta,
                        "spread_adf_p": adf_pvalue(spread), "half_life": half_life(spread)})
    return results


def run_selection_funnel(log_train: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Pasa cada grupo candidato por el embudo y registra dónde se descartó.

    Args:
        log_train: log-precios de la partición de entrenamiento.
        settings: umbrales del embudo (ver config.COINTEGRATION).

    Returns:
        Tabla con una fila por grupo: estadísticos de cada etapa, la etapa de
        descarte ("aprobado" si pasó todo) y el vector beta normalizado.
    """
    rolling_corr = rolling_pair_correlation(log_train, settings["corr_window"])
    k_ar_diff = select_k_ar_diff(log_train, settings["max_lags"])
    n_groups = len(candidate_groups(list(log_train.columns)))
    rows = []

    for group in candidate_groups(list(log_train.columns)):
        data = log_train[list(group)]
        row = {"group": "-".join(group), "size": len(group), "k_ar_diff": k_ar_diff}
        row["corr_mean"], row["corr_min"] = group_correlation(rolling_corr, group)
        row.update(engle_granger_group(data, group))
        row["eg_pass_bonferroni"] = row["eg_pvalue"] < settings["alpha"] / n_groups

        joh = johansen_test(data, k_ar_diff)
        row.update({"rank_trace": joh["rank_trace"], "rank_max_eig": joh["rank_max_eig"],
                    "eig_1": float(joh["eigenvalues"][0]),
                    "trace_r0": float(joh["trace"][0]), "trace_cv95_r0": float(joh["trace_cv95"][0]),
                    "max_eig_r0": float(joh["max_eig"][0]), "max_eig_cv95_r0": float(joh["max_eig_cv95"][0])})

        rank = max(joh["rank_trace"], 1)
        vectors = evaluate_vectors(data, joh, rank)
        best = min(vectors, key=lambda v: v["spread_adf_p"])
        row.update({"vector_column": best["column"], "spread_adf_p": best["spread_adf_p"],
                    "half_life": best["half_life"],
                    "beta": {a: round(float(b), 6) for a, b in zip(group, best["beta"])}})
        row["stage_discarded"] = discard_stage(row, settings)
        rows.append(row)

    return pd.DataFrame(rows).set_index("group")


def discard_stage(row: dict, settings: dict) -> str:
    """Primera etapa del embudo que el grupo no supera."""
    hl_min, hl_max = settings["half_life_range"]
    if row["corr_mean"] < settings["min_corr"]:
        return "2-correlación"
    if row["eg_pvalue"] >= settings["alpha"]:
        return "3-Engle-Granger"
    if row["rank_trace"] < 1:
        return "4-Johansen"
    if row["spread_adf_p"] >= settings["alpha"] or not hl_min <= row["half_life"] <= hl_max:
        return "5-validación del spread"
    return "aprobado"


def pick_final(table: pd.DataFrame) -> tuple[str, str | None]:
    """Elige el grupo final y el mejor par para comparar (aclaración del profesor).

    Entre los grupos aprobados de 3 o más activos se elige el más reversivo: el
    de menor vida media (corrección de error más rápida). Si ninguno de 3 o más
    activos aprueba, se usa el mejor par.

    Returns:
        (grupo final, mejor par) o (grupo final, None) si no hay par aprobado.
    """
    approved = table[table["stage_discarded"] == "aprobado"]
    if approved.empty:
        raise RuntimeError("Ningún grupo superó el embudo de selección")
    multi = approved[approved["size"] >= 3]
    pairs = approved[approved["size"] == 2]
    best_pair = pairs["half_life"].idxmin() if not pairs.empty else None
    final = multi["half_life"].idxmin() if not multi.empty else best_pair
    return final, best_pair
