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
from cointegration.johansen import johansen_table, johansen_test
from cointegration.selection import pick_final, run_selection_funnel
from data.loader import load_prices
from data.validation import clean_prices, split_data, validate_prices
from cointegration.stability import rolling_johansen, stationarity_by_split
from kalman.diagnostics import innovation_diagnostics
from kalman.pipeline import order_assets, run_filters
from reporting.figures import plot_kalman_gain, plot_prices, plot_rolling_eigenvector, plot_weights
from reporting.worked_example import worked_example_filter1, worked_example_filter2


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

    # Tabla completa de traza y máximo eigenvalor del grupo final y del mejor par (paso 4)
    details = []
    for label, group in (("grupo_final", final), ("mejor_par", best_pair)):
        if group is not None:
            joh = johansen_test(log_train[group.split("-")], vector["k_ar_diff"])
            details.append(johansen_table(joh).assign(grupo=group, rol=label))
    pd.concat(details).to_csv(out / "johansen_detail.csv", index=False, float_format="%.6f")

    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print(table[["corr_mean", "eg_pvalue", "rank_trace", "spread_adf_p",
                     "half_life", "stage_discarded"]].round(4))
    print(f"  Grupos evaluados: {len(table)} | Final: {final} | Mejor par: {best_pair}")
    print(f"  Vector beta: {vector['beta']}")
    return vector


def split_labels(splits: dict) -> np.ndarray:
    """Etiqueta de partición ("train", "test", "val") para cada fecha."""
    return np.concatenate([[name] * len(df) for name, df in splits.items()])


def run_kalman_stage(prices: pd.DataFrame, splits: dict, vector: dict) -> pd.DataFrame:
    """Etapa 3: filtro 1 (pesos dinámicos) y filtro 2 (media, varianza y z-score)."""
    print("[3/6] Filtros de Kalman")
    k = config.KALMAN
    assets, beta0 = order_assets(vector)
    f1, f2, params = run_filters(np.log(prices[assets]), beta0, len(splits["train"]), k)

    f2.insert(0, "split", split_labels(splits))
    f2.insert(1, "tradable", np.arange(len(f2)) >= k["burn_in"])
    diagnostics = {name: innovation_diagnostics(f2.loc[(f2["split"] == name) & f2["tradable"], "innovation_norm"])
                   for name in ("train", "test", "val")}
    params = {"assets_order": assets, **params, "burn_in_days": k["burn_in"],
              "burn_in_end": str(f2.index[k["burn_in"]].date()), "innovation_diagnostics": diagnostics}

    f1.to_csv(config.ARTIFACTS_DIR / "filter1.csv", float_format="%.10g")
    f2.to_csv(config.ARTIFACTS_DIR / "filter2.csv", float_format="%.10g")
    with open(config.ARTIFACTS_DIR / "kalman_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    ar = params["filter2"]
    print(f"  Filtro 1: Q = {k['q_weights']:.0e} I, R = {params['filter1']['r']:.2e}, burn-in hasta {params['burn_in_end']}")
    print(f"  Filtro 2: phi = {ar['phi']:.4f} (vida media {ar['half_life']:.1f} días), Q = {ar['q']:.0e} por MV")
    for name, d in diagnostics.items():
        print(f"  Innovaciones {name:>5}: varianza {d['variance']:.2f}, acf1 {d['acf1']:+.3f}, Ljung-Box p {d['ljung_box_p']:.3f}")
    return f1


def run_stability_stage(prices: pd.DataFrame, splits: dict, vector: dict, f1: pd.DataFrame) -> None:
    """Johansen móvil y estacionariedad del spread estático y dinámico por partición."""
    print("  Estabilidad de la relación")
    assets, beta0 = order_assets(vector)
    log_prices = np.log(prices[assets])
    rolling = rolling_johansen(log_prices, config.ROLLING["window"], config.ROLLING["step"], vector["k_ar_diff"])
    rolling.to_csv(config.ARTIFACTS_DIR / "johansen_rolling.csv", float_format="%.6f")

    labels = pd.Series(split_labels(splits), index=prices.index)
    spreads = {"estatico": log_prices @ beta0, "kalman": f1["spread"].iloc[config.KALMAN["burn_in"]:]}
    table = stationarity_by_split(spreads, labels)
    table.to_csv(config.ARTIFACTS_DIR / "stability_oos.csv", index=False, float_format="%.6f")

    print(f"  Johansen móvil ({config.ROLLING['window']} días): cointegra en "
          f"{rolling['cointegrated'].mean():.0%} de {len(rolling)} ventanas")
    with pd.option_context("display.width", 120):
        print(table.round(4).to_string(index=False))


def run_figures_stage(prices: pd.DataFrame, splits: dict, vector: dict) -> None:
    """Figuras del reporte (sección 3.10) a partir de los artifacts."""
    print("[+] Figuras del reporte")
    out, folder = config.ARTIFACTS_DIR, config.DOCS_DIR / "figures"
    starts = {name: df.index[0] for name, df in splits.items()}
    filter1 = pd.read_csv(out / "filter1.csv", index_col=0, parse_dates=True)
    filter2 = pd.read_csv(out / "filter2.csv", index_col=0, parse_dates=True)
    rolling = pd.read_csv(out / "johansen_rolling.csv", index_col=0, parse_dates=True)
    with open(out / "kalman_params.json", encoding="utf-8") as f:
        burn_in_end = json.load(f)["burn_in_end"]

    paths = [plot_prices(prices[vector["assets"]], starts, folder),
             plot_weights(filter1, vector["beta"], starts, folder),
             plot_rolling_eigenvector(rolling, starts, config.ROLLING["window"], folder),
             plot_kalman_gain(filter1, filter2, starts, burn_in_end, folder)]
    print(f"  {len(paths)} figuras en {folder.relative_to(config.ROOT)}")

    # Ejemplo numérico trabajado (sección 3.5): días alrededor de una entrada en prueba
    with open(out / "kalman_params.json", encoding="utf-8") as f:
        params = json.load(f)
    dates = pd.DatetimeIndex(config.WORKED_EXAMPLE_DATES)
    log_prices = np.log(prices[params["assets_order"]])
    example1 = worked_example_filter1(log_prices, filter1, params["filter1"]["q"], dates)
    example2 = worked_example_filter2(filter1["spread"], filter2, params["filter2"], dates)
    example1.to_csv(out / "worked_example_filter1.csv", float_format="%.10g")
    example2.to_csv(out / "worked_example_filter2.csv", float_format="%.10g")
    worst = max(example1["error de reconstruccion"].max(), example2["error de reconstruccion"].max())
    print(f"  Ejemplo numérico: {len(dates)} días, error máximo de reconstrucción {worst:.1e}")


def main() -> None:
    """Ejecuta todas las etapas del pipeline en orden."""
    np.random.seed(config.SEED)
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Lab03 MyST — Equipo 7 | semilla = {config.SEED}")
    prices, splits = run_data_stage()
    vector = run_cointegration_stage(splits)
    f1 = run_kalman_stage(prices, splits, vector)
    run_stability_stage(prices, splits, vector, f1)
    run_figures_stage(prices, splits, vector)


if __name__ == "__main__":
    main()
