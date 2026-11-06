"""Parámetros globales del laboratorio (sección 3.1 de los lineamientos).

Todos los módulos leen sus parámetros de aquí para que el pipeline sea
reproducible desde un único lugar.
"""
from pathlib import Path

# --- Reproducibilidad -------------------------------------------------------
SEED = 42  # Semilla fija para numpy (documentada en el README)

# --- Rutas ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

# --- Datos ------------------------------------------------------------------
# Universo: utilities eléctricas reguladas de EE. UU. (sector XLU). Sus utilidades
# las fija el regulador (ROE autorizado) y se comportan como bonos, así que comparten
# un factor común: las tasas de interés. Se eligió tras evaluar 6 universos de un
# solo sector con el mismo embudo (ver cointegration/universe_search.py).
TICKERS = ["DUK", "SO", "AEP", "XEL"]
START_DATE = "2011-11-01"        # 15 años de datos diarios
END_DATE = "2026-11-01"          # Fecha exclusiva: el último día es el 30-oct-2026
PRICES_FILE = DATA_DIR / "prices.csv"   # Copia local versionada; evita depender de Yahoo
SPLIT = (0.60, 0.20, 0.20)       # Entrenamiento / prueba / validación (cronológico)
MAX_FFILL_DAYS = 3               # Huecos aislados de hasta 3 días se rellenan hacia adelante

# --- Selección por cointegración (sección 3.2) -----------------------------
COINTEGRATION = {
    "alpha": 0.05,               # Nivel de significancia de ADF, Engle-Granger y Johansen
    "min_corr": 0.70,            # Filtro heurístico de correlación (promedio del grupo)
    "corr_window": 252,          # Ventana de la correlación móvil (1 año hábil)
    "max_lags": 10,              # Rezagos máximos al elegir el VECM por BIC
    "half_life_range": (5, 60),  # Vida media operable en días (S14)
}

# --- Filtros de Kalman (secciones 3.3 y 3.4) ------------------------------
KALMAN = {
    "q_weights": 1e-8,           # Q del filtro 1: los pesos cambian despacio (ver README)
    "p0_weights": 1e-6,          # P_0 del filtro 1: alta confianza en el vector de Johansen
    "burn_in": 60,               # Días descartados antes de operar (~3 vidas medias)
    "q_mean_grid": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Q del filtro 2: por máxima verosimilitud
    "p0_mean": 1e-4,             # P_0 del filtro 2
    "ewma_lambda": 0.97,         # Persistencia del EWMA de la varianza del spread
}

# --- Johansen móvil (figura 4 y estabilidad) ------------------------------
ROLLING = {"window": 756, "step": 21}   # Ventanas de 3 años que avanzan 1 mes

# --- Ejemplo numérico de los filtros (sección 3.5) --------------------------
# Días consecutivos en prueba en los que el z-score cruza -2 (señal de entrada larga)
WORKED_EXAMPLE_DATES = ["2022-09-22", "2022-09-23", "2022-09-26", "2022-09-27", "2022-09-28"]

# --- Búsqueda del universo (documentada para el control de pruebas múltiples) ---
# Cada universo es un solo sector con un factor económico común. Todos se evalúan
# con el mismo embudo y solo con la partición de entrenamiento; el reporte incluye
# todos los grupos probados en todos los universos.
CANDIDATE_UNIVERSES = {
    "petroleras": (["XOM", "CVX", "COP", "EOG"], "Precio del crudo"),
    "utilities": (["DUK", "SO", "AEP", "XEL"], "Tarifas reguladas y tasas de interés"),
    "consumo_basico": (["KO", "PEP", "PG", "CL"], "Demanda inelástica y márgenes estables"),
    "ferrocarriles": (["UNP", "CSX", "NSC", "CNI"], "Volumen de carga en Norteamérica"),
    "bancos": (["JPM", "BAC", "C", "WFC"], "Curva de tasas y ciclo de crédito"),
    "etf_tecnologia": (["XLK", "VGT", "IYW", "IGM"], "Mismas empresas tecnológicas subyacentes"),
}

# --- Señales (sección 3.6) --------------------------------------------------
# Valores iniciales; z_entrada y z_salida se recalibran con grid search en entrenamiento.
SIGNALS = {
    "z_entry": 2.0,              # Entrada: |z| > 2 desviaciones del equilibrio
    "z_exit": 0.5,               # Salida: el spread regresó cerca de su media
    "z_stop": 4.0,               # Stop: el spread se alejó el doble del umbral de entrada
    "max_holding_multiple": 3.0, # Tenencia máxima = 3 vidas medias (el spread debió revertir ~87.5%)
}
# La vida media es la del spread que se OPERA (spread de Kalman, AR(1) del filtro 2:
# kalman_params.json -> filter2.half_life, 15.5 días -> 47 días), no la del spread
# estático de Johansen (24 días), que no es el que genera las señales.

# --- Grid search de umbrales (sección 3.6), solo con entrenamiento ---------
GRID = {
    "z_entry": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "z_exit": [0.0, 0.25, 0.5, 0.75],
    "max_holding_multiple": [2.0, 3.0, 4.0],
    "z_stop": 4.0,               # Fijo: es un límite de riesgo, no un parámetro a optimizar
    "min_trades": 15,            # Mínimo de operaciones en entrenamiento para competir
}

# --- Costos y capital -------------------------------------------------------
COMMISSION = 0.00125             # 0.125% por pata, a la entrada y a la salida
BORROW_RATE_ANNUAL = 0.0025      # 0.25% anual sobre el nocional corto
TRADING_DAYS = 252               # Devengo diario = tasa anual / 252
INITIAL_CAPITAL = 1_000_000
CAPITAL_FRACTION = 0.80          # 80% del capital desplegado en el spread
