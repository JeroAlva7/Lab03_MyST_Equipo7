"""Configuración de pytest: rutas y datos compartidos por las pruebas."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  (requiere el path de arriba)
from data.validation import split_data  # noqa: E402


@pytest.fixture(scope="session")
def prices() -> pd.DataFrame:
    """Precios versionados en el repositorio (no se descarga nada)."""
    return pd.read_csv(config.PRICES_FILE, index_col="date", parse_dates=True)[config.TICKERS]


@pytest.fixture(scope="session")
def n_train(prices) -> int:
    """Número de fechas de entrenamiento según la partición 60/20/20."""
    return len(split_data(prices, config.SPLIT)["train"])


@pytest.fixture(scope="session")
def vector() -> dict:
    """Vector cointegrante seleccionado por el pipeline."""
    with open(config.ARTIFACTS_DIR / "johansen_vector.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def log_prices(prices, vector) -> pd.DataFrame:
    """Log-precios con la pata normalizada primero."""
    from kalman.pipeline import order_assets
    assets, _ = order_assets(vector)
    return np.log(prices[assets])
