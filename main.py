"""Pipeline completo del Laboratorio 03. Se ejecuta con: python main.py

Etapas (cada una vive en su propio paquete):
    1. data/           Descarga, limpieza y validación de precios.
    2. cointegration/  Screening, Engle-Granger y Johansen sobre entrenamiento.
    3. kalman/         Filtro 1 (pesos beta_t) y filtro 2 (media/varianza del spread).
    4. strategy/       VECM, z-score causal, señales y sizing.
    5. backtest/       Simulación con costos y walk-forward.
    6. performance/    Métricas y escritura de artifacts/ para el dashboard.
"""
import json

import numpy as np
import pandas as pd

import config
from cointegration.correlation import rolling_pair_correlation
from cointegration.integration import all_integrated_of_order_one, integration_order_table
from cointegration.selection import pick_final, run_selection_funnel
from data.loader import load_prices
from data.validation import clean_prices, split_data, validate_prices


def run_data_stage() -> dict:
    """Etapa 1: carga, limpia, valida y parte los precios."""
    print("[1/6] Datos")
    raw = load_prices(config.TICKERS, config.START_DATE, config.END_DATE, config.PRICES_FILE)
    prices, report = clean_prices(raw, config.MAX_FFILL_DAYS)
    validate_prices(prices)
    splits = split_data(prices, config.SPLIT)

    report["splits"] = {
        name: {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "rows": len(df)}
        for name, df in splits.items()
    }
    prices.to_csv(config.ARTIFACTS_DIR / "prices.csv", float_format="%.6f")
    with open(config.ARTIFACTS_DIR / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"  {report['rows_clean']} días hábiles, {report['start']} a {report['end']}")
    for name, info in report["splits"].items():
        print(f"  {name:>5}: {info['start']} a {info['end']} ({info['rows']} días)")
    return splits


def run_cointegration_stage(splits: dict) -> dict:
    """Etapa 2: orden de integración, embudo de selección y vector final (solo entrenamiento)."""
    print("[2/6] Cointegración (solo entrenamiento)")
    log_train = np.log(splits["train"])
    out = config.ARTIFACTS_DIR

    integration = integration_order_table(log_train, config.COINTEGRATION["alpha"])
    integration.to_csv(out / "integration_tests.csv", float_format="%.6f")
    print(f"  Todos los log-precios son I(1): {all_integrated_of_order_one(integration)}")

    rolling_pair_correlation(log_train, config.COINTEGRATION["corr_window"]).to_csv(
        out / "rolling_correlation.csv", float_format="%.4f")

    table = run_selection_funnel(log_train, config.COINTEGRATION)
    table.to_csv(out / "selection_table.csv", float_format="%.6f")
    final, best_pair = pick_final(table)
    chosen = table.loc[final]

    vector = {
        "group": final, "assets": final.split("-"),
        "beta": {k: float(v) for k, v in chosen["beta"].items()},
        "spread_adf_p": float(chosen["spread_adf_p"]), "half_life": float(chosen["half_life"]),
        "eigenvalue_1": float(chosen["eig_1"]), "rank_trace": int(chosen["rank_trace"]),
        "k_ar_diff": int(chosen["k_ar_diff"]), "best_pair": best_pair,
        "groups_tested": int(len(table)),
    }
    with open(out / "johansen_vector.json", "w", encoding="utf-8") as f:
        json.dump(vector, f, indent=2)

    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print(table[["corr_mean", "eg_pvalue", "rank_trace", "spread_adf_p",
                     "half_life", "stage_discarded"]].round(4))
    print(f"  Grupos evaluados: {len(table)} | Final: {final} | Mejor par: {best_pair}")
    print(f"  Vector beta: {vector['beta']}")
    return vector


def main() -> None:
    """Ejecuta todas las etapas del pipeline en orden."""
    np.random.seed(config.SEED)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Lab03 MyST — Equipo 7 | semilla = {config.SEED}")
    splits = run_data_stage()
    run_cointegration_stage(splits)


if __name__ == "__main__":
    main()
