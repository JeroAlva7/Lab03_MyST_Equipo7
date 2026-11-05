"""Figuras obligatorias del reporte (sección 3.10), generadas desde los artifacts.

Convenciones para que sean legibles en el reporte y en la presentación:
título, ejes con unidades, leyenda, líneas finas, cuadrícula tenue, divisiones de
entrenamiento / prueba / validación marcadas y un color fijo por activo.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Genera archivos sin abrir ventanas
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Un color fijo por activo (paleta validada para daltonismo); el gris es neutro
ASSET_COLORS = {"DUK": "#2a78d6", "SO": "#eb6834", "AEP": "#1baf7a", "XEL": "#eda100"}
NEUTRAL = "#6b6b6b"
SPLIT_NAMES = {"train": "Entrenamiento", "test": "Prueba", "val": "Validación"}

plt.rcParams.update({
    "figure.figsize": (11, 5.5), "figure.dpi": 150, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def mark_splits(ax: plt.Axes, split_starts: dict) -> None:
    """Dibuja el inicio de prueba y validación y rotula cada partición dentro del eje."""
    x_min = ax.get_xlim()[0]
    for name, start in split_starts.items():
        if name != "train":
            ax.axvline(start, color=NEUTRAL, linestyle="--", linewidth=0.9)
        x = max(mdates.date2num(start), x_min)
        ax.annotate(SPLIT_NAMES[name], xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -12), textcoords="offset points", fontsize=9, color=NEUTRAL)


def save(fig: plt.Figure, folder: Path, name: str) -> Path:
    """Guarda la figura como PNG y libera memoria."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prices(prices: pd.DataFrame, split_starts: dict, folder: Path) -> Path:
    """Figura 1: precios ajustados de cada activo del grupo."""
    fig, ax = plt.subplots()
    for asset in prices.columns:
        ax.plot(prices.index, prices[asset], color=ASSET_COLORS.get(asset), label=asset)
    mark_splits(ax, split_starts)
    ax.set(title="Figura 1. Precios ajustados de las utilities del grupo",
           xlabel="Fecha", ylabel="Precio ajustado por dividendos (USD)")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    return save(fig, folder, "fig01_precios.png")


def plot_weights(filter1: pd.DataFrame, beta0: dict, split_starts: dict, folder: Path) -> Path:
    """Figura 3: pesos dinámicos del filtro 1, un panel por activo (DUK está fijo en +1)."""
    free = [a for a in beta0 if abs(beta0[a] - 1.0) > 1e-9]
    fig, axes = plt.subplots(len(free), 1, sharex=True, figsize=(11, 2.6 * len(free)))
    for ax, asset in zip(axes, free):
        ax.plot(filter1.index, filter1[f"beta_{asset}"], color=ASSET_COLORS.get(asset),
                label=rf"$\hat{{\beta}}$ {asset} (Kalman)")
        ax.axhline(beta0[asset], color=NEUTRAL, linestyle=":", linewidth=1.1,
                   label=f"β {asset} Johansen = {beta0[asset]:.3f}")
        ax.set_ylabel(f"Peso de {asset}")
        ax.legend(loc="lower left", fontsize=9)
        mark_splits(ax, split_starts)
    axes[0].set_title(r"Figura 3. Evolución del vector de pesos $\hat{\beta}_t$ (filtro de Kalman 1; DUK normalizado en +1)")
    axes[-1].set_xlabel("Fecha")
    return save(fig, folder, "fig03_pesos.png")


def plot_rolling_eigenvector(rolling: pd.DataFrame, split_starts: dict, window: int, folder: Path) -> Path:
    """Figura 4: primer eigenvector de Johansen móvil y su estadístico de traza."""
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 7.5),
                                      gridspec_kw={"height_ratios": [3, 2]})
    for column in [c for c in rolling.columns if c.startswith("evec_")]:
        asset = column.removeprefix("evec_")
        top.plot(rolling.index, rolling[column], color=ASSET_COLORS.get(asset), label=asset,
                 marker="o", markersize=3)
    top.axhline(0, color=NEUTRAL, linewidth=0.8)
    top.set(ylabel="Componente (norma 1, DUK > 0)",
            title=f"Figura 4. Primer eigenvector de Johansen móvil (ventanas de {window} días)")
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    mark_splits(top, split_starts)

    bottom.plot(rolling.index, rolling["trace"], color="#2a78d6", label="Traza (r₀ = 0)")
    bottom.plot(rolling.index, rolling["trace_cv95"], color=NEUTRAL, linestyle="--",
                label="Valor crítico 95%")
    bottom.fill_between(rolling.index, 0, rolling["trace"].max() * 1.05,
                        where=rolling["cointegrated"].astype(bool), color="#1baf7a", alpha=0.12,
                        label="Ventana cointegrada")
    bottom.set(ylabel="Estadístico de traza", xlabel="Fecha de cierre de la ventana")
    bottom.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return save(fig, folder, "fig04_eigenvector_movil.png")


def plot_kalman_gain(filter1: pd.DataFrame, filter2: pd.DataFrame, split_starts: dict,
                     burn_in_end: str, folder: Path) -> Path:
    """Figura 5: ganancia de Kalman de ambos filtros (con qué agresividad se actualizan)."""
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
    for column in [c for c in filter1.columns if c.startswith("gain_") and c != "gain_intercept"]:
        asset = column.removeprefix("gain_")
        top.plot(filter1.index, filter1[column].abs(), color=ASSET_COLORS.get(asset),
                 label=rf"|K| de $\beta$ {asset}")
    top.set(title="Figura 5. Ganancia de Kalman de ambos filtros (magnitud)",
            ylabel="Filtro 1: |ganancia| por peso")
    top.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    mark_splits(top, split_starts)

    bottom.plot(filter2.index, filter2["gain"], color="#2a78d6", label=r"K de la media $\mu_t$")
    bottom.axvline(pd.Timestamp(burn_in_end), color="#e34948", linestyle=":", linewidth=1,
                   label="Fin del burn-in")
    bottom.set(ylabel=r"Filtro 2: ganancia de $\mu_t$ (escala log)", xlabel="Fecha")
    bottom.set_yscale("log")
    bottom.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    return save(fig, folder, "fig05_ganancia_kalman.png")
