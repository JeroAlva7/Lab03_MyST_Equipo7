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
from kalman.diagnostics import innovation_diagnostics
from kalman.spread_filter import fit_ar1, log_likelihood, run_spread_filter
from kalman.weights_filter import run_weights_filter


def run_data_stage() -> tuple[pd.DataFrame, dict]:
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
    return prices, splits


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


def run_kalman_stage(prices: pd.DataFrame, splits: dict, vector: dict) -> None:
    """Etapa 3: filtro 1 (pesos dinámicos) y filtro 2 (media, varianza y z-score)."""
    print("[3/6] Filtros de Kalman")
    k = config.KALMAN
    # La pata normalizada (beta = +1) va primero: es la observación del filtro 1
    assets = sorted(vector["assets"], key=lambda a: abs(vector["beta"][a] - 1.0) > 1e-9)
    log_prices = np.log(prices[assets])
    beta0 = np.array([vector["beta"][a] for a in assets])
    n_train = len(splits["train"])

    # R del filtro 1: varianza del spread de Johansen en entrenamiento ("price noise")
    johansen_spread = log_prices.iloc[:n_train] @ beta0
    r_weights = float(johansen_spread.var())
    f1 = run_weights_filter(log_prices, beta0, float(johansen_spread.mean()),
                            k["q_weights"], r_weights, k["p0_weights"])

    # Filtro 2: AR(1) del spread y Q por máxima verosimilitud, solo en entrenamiento
    train_spread = f1["spread"].iloc[k["burn_in"]:n_train]
    ar = fit_ar1(train_spread)
    likelihoods = {q: log_likelihood(train_spread, ar["phi"], ar["mu"], q, ar["resid_var"], k["p0_mean"])
                   for q in k["q_mean_grid"]}
    q_mean = max(likelihoods, key=likelihoods.get)
    f2 = run_spread_filter(f1["spread"], ar["phi"], ar["mu"], q_mean, ar["resid_var"],
                           k["p0_mean"], k["ewma_lambda"])

    labels = np.array(["train"] * n_train + ["test"] * len(splits["test"]) + ["val"] * len(splits["val"]))
    f2.insert(0, "split", labels)
    f2.insert(1, "tradable", np.arange(len(f2)) >= k["burn_in"])
    diagnostics = {name: innovation_diagnostics(f2.loc[(f2["split"] == name) & f2["tradable"], "innovation_norm"])
                   for name in ("train", "test", "val")}

    params = {"assets_order": assets, "filter1": {"q": k["q_weights"], "r": r_weights, "p0": k["p0_weights"]},
              "burn_in_days": k["burn_in"], "burn_in_end": str(f2.index[k["burn_in"]].date()),
              "filter2": {**ar, "q": q_mean, "p0": k["p0_mean"], "ewma_lambda": k["ewma_lambda"],
                          "log_likelihood_grid": {str(q): v for q, v in likelihoods.items()}},
              "innovation_diagnostics": diagnostics}
    f1.to_csv(config.ARTIFACTS_DIR / "filter1.csv", float_format="%.10g")
    f2.to_csv(config.ARTIFACTS_DIR / "filter2.csv", float_format="%.10g")
    with open(config.ARTIFACTS_DIR / "kalman_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"  Filtro 1: Q = {k['q_weights']:.0e} I, R = {r_weights:.2e}, burn-in hasta {params['burn_in_end']}")
    print(f"  Filtro 2: phi = {ar['phi']:.4f} (vida media {ar['half_life']:.1f} días), Q = {q_mean:.0e} por MV")
    for name, d in diagnostics.items():
        print(f"  Innovaciones {name:>5}: varianza {d['variance']:.2f}, acf1 {d['acf1']:+.3f}, Ljung-Box p {d['ljung_box_p']:.3f}")


def main() -> None:
    """Ejecuta todas las etapas del pipeline en orden."""
    np.random.seed(config.SEED)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Lab03 MyST — Equipo 7 | semilla = {config.SEED}")
    prices, splits = run_data_stage()
    vector = run_cointegration_stage(splits)
    run_kalman_stage(prices, splits, vector)


if __name__ == "__main__":
    main()
