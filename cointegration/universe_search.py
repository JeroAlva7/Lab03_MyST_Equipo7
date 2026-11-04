"""Búsqueda documentada del universo de activos (pasos 1 y 6 de la sección 3.2).

Aplica el mismo embudo de selección a cada universo candidato de config.py y
guarda todos los grupos probados, para reportar el total de pruebas realizadas.

Uso:  python -m cointegration.universe_search
"""
import numpy as np
import pandas as pd

import config
from cointegration.selection import run_selection_funnel
from data.loader import load_prices
from data.validation import clean_prices, split_data, validate_prices


def evaluate_universe(name: str, tickers: list[str]) -> pd.DataFrame:
    """Descarga (o lee del CSV) un universo y lo pasa por el embudo con datos de entrenamiento."""
    cache = config.DATA_DIR / "universes" / f"{name}.csv"
    raw = load_prices(tickers, config.START_DATE, config.END_DATE, cache)
    prices, _ = clean_prices(raw, config.MAX_FFILL_DAYS)
    validate_prices(prices)
    log_train = np.log(split_data(prices, config.SPLIT)["train"])
    table = run_selection_funnel(log_train, config.COINTEGRATION)
    table.insert(0, "universe", name)
    return table


def main() -> None:
    """Evalúa todos los universos candidatos y guarda la tabla completa."""
    tables = []
    for name, (tickers, factor) in config.CANDIDATE_UNIVERSES.items():
        print(f"- {name} ({', '.join(tickers)}): {factor}")
        tables.append(evaluate_universe(name, tickers))
    results = pd.concat(tables)
    results.to_csv(config.ARTIFACTS_DIR / "universe_search.csv", float_format="%.6f")

    columns = ["universe", "corr_mean", "eg_pvalue", "rank_trace", "trace_r0",
               "trace_cv95_r0", "spread_adf_p", "half_life", "stage_discarded"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(results[columns].round(4))
    print(f"\nGrupos evaluados en total: {len(results)}")
    print(results.groupby("universe")["stage_discarded"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
