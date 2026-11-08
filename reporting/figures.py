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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

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


def plot_zscore(signals: pd.DataFrame, params: dict, split_starts: dict, folder: Path) -> Path:
    """Figura 6: z-score prior del filtro 2 con los umbrales elegidos en entrenamiento.

    Se sombrean los días con posición abierta (azul: spread largo; naranja: spread corto).
    """
    z, direction = signals["z"], signals["direction"]
    fig, ax = plt.subplots()
    ax.plot(z.index, z, color=NEUTRAL, linewidth=0.7, label="z-score (prior, causal)")
    levels = [(params["z_entry"], "#2a78d6", "-", f"Entrada ±{params['z_entry']:g}"),
              (params["z_stop"], "#e34948", "--", f"Stop ±{params['z_stop']:g}")]
    for level, color, style, label in levels:
        ax.axhline(level, color=color, linestyle=style, linewidth=1, label=label)
        ax.axhline(-level, color=color, linestyle=style, linewidth=1)
    exit_label = f"Salida ±{params['z_exit']:g}" if params["z_exit"] else "Salida: cruce de 0"
    ax.axhline(params["z_exit"], color="#1baf7a", linewidth=1, label=exit_label)
    if params["z_exit"]:
        ax.axhline(-params["z_exit"], color="#1baf7a", linewidth=1)
    bottom, top = -params["z_stop"] * 1.5, params["z_stop"] * 1.5
    for side, color, label in [(1, "#2a78d6", "Spread largo"), (-1, "#eb6834", "Spread corto")]:
        ax.fill_between(z.index, bottom, top, where=(direction == side).to_numpy(), color=color,
                        alpha=0.10, linewidth=0, label=label)
    ax.set_ylim(bottom, top)
    mark_splits(ax, split_starts)
    ax.set(title="Figura 6. z-score del spread con umbrales de entrada, salida y stop",
           xlabel="Fecha", ylabel=r"$z_t = (S_t - \mu_{t|t-1}) / \sigma_{t|t-1}$")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12), fontsize=9)
    return save(fig, folder, "fig06_zscore.png")


def plot_grid_robustness(grid: pd.DataFrame, chosen: dict, min_trades: int, folder: Path) -> Path:
    """Figura A1: Sharpe de entrenamiento de cada combinación del grid (robustez de la elección).

    Un panel por múltiplo de tenencia máxima; color divergente centrado en 0 (azul gana, rojo
    pierde). Las celdas tachadas no compiten por tener menos de `min_trades` operaciones.
    """
    multiples = sorted(grid["max_holding_multiple"].unique())
    limit = float(np.nanmax(np.abs(grid["sharpe"])))
    cmap = LinearSegmentedColormap.from_list("sharpe", ["#e34948", "#f2f2f2", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, len(multiples), figsize=(4.2 * len(multiples) + 1, 4.8), sharey=True)
    for ax, multiple in zip(np.atleast_1d(axes), multiples):
        panel = grid[grid["max_holding_multiple"] == multiple]
        sharpe = panel.pivot(index="z_entry", columns="z_exit", values="sharpe")
        trades = panel.pivot(index="z_entry", columns="z_exit", values="trades")
        image = ax.imshow(sharpe.to_numpy(), cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for i, z_in in enumerate(sharpe.index):
            for j, z_out in enumerate(sharpe.columns):
                few = trades.iloc[i, j] < min_trades
                if few:  # Se atenúa la celda y se tacha: no compite en la elección
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                               alpha=0.55, linewidth=0))
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//",
                                               edgecolor="#c4c4c4", linewidth=0))
                ax.text(j, i, f"{sharpe.iloc[i, j]:.2f}\n{int(trades.iloc[i, j])} op",
                        ha="center", va="center", fontsize=8, color="#6b6b6b" if few else "#1a1a1a")
                if (z_in, z_out, multiple) == (chosen["z_entry"], chosen["z_exit"],
                                               chosen["max_holding_multiple"]):
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor="#1a1a1a", linewidth=2.2))
        ax.set_xticks(range(len(sharpe.columns)), [f"{v:g}" for v in sharpe.columns])
        ax.set_yticks(range(len(sharpe.index)), [f"{v:g}" for v in sharpe.index])
        ax.set_title(f"Tenencia máxima = {multiple:g} × vida media", fontsize=11)
        ax.set_xlabel("z de salida")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("z de entrada")
    fig.colorbar(image, ax=axes, label="Sharpe anualizado (entrenamiento)", shrink=0.85)
    fig.suptitle("Figura A1. Robustez del grid search: Sharpe de entrenamiento por combinación\n"
                 f"(recuadro negro = elegida; tachado = menos de {min_trades} operaciones, no compite)",
                 fontsize=12, fontweight="bold", y=1.06)
    return save(fig, folder, "figA1_grid_robustez.png")


SPLIT_COLORS = {"train": "#2a78d6", "test": "#eb6834", "val": "#1baf7a"}


def plot_spread_signals(spread: pd.Series, signals: pd.DataFrame, split_starts: dict, folder: Path) -> Path:
    """Figura 2: spread del filtro 1 con las señales de entrada y salida (fecha de decisión)."""
    fig, ax = plt.subplots()
    ax.plot(spread.index, spread, color=NEUTRAL, linewidth=0.7, label=r"Spread $S_t$")
    markers = [("entrada larga", "^", "#1baf7a", "Entrada larga"), ("entrada corta", "v", "#e34948", "Entrada corta"),
               ("salida", "o", "#2a78d6", "Salida por reversión"), ("tiempo", "s", "#eda100", "Salida por tiempo"),
               ("stop", "X", "#e34948", "Stop")]
    for event, marker, color, label in markers:
        days = signals.index[signals["event"] == event]
        if len(days):
            ax.scatter(days, spread[days], marker=marker, s=45, color=color, edgecolor="white",
                       linewidth=0.6, zorder=3, label=f"{label} ({len(days)})")
    mark_splits(ax, split_starts)
    ax.set(title="Figura 2. Spread del portafolio con las señales de entrada y salida",
           xlabel="Fecha (día de la señal; se ejecuta al cierre del día siguiente)",
           ylabel=r"$S_t = \hat{\beta}_t^\top \ln P_t$")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14), fontsize=9)
    return save(fig, folder, "fig02_spread_senales.png")


def plot_equity_curves(equity: pd.DataFrame, summary: dict, split_starts: dict, folder: Path) -> Path:
    """Figura 7: curva de capital neta de costos por partición, con el drawdown debajo."""
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(11, 7),
                                      gridspec_kw={"height_ratios": [3, 1.2]})
    for name, label in SPLIT_NAMES.items():
        mask = (equity["split"] == name).to_numpy(copy=True)
        mask[max(mask.argmax() - 1, 0)] = True          # une el tramo con el anterior
        card = summary[name]
        top.plot(equity.index[mask], equity["equity"][mask] / 1e6, color=SPLIT_COLORS[name],
                 label=f"{label}: {card['return']:+.1%}, Sharpe {card['sharpe']:.2f}, "
                       f"{card['trades']} operaciones")
    top.axhline(1.0, color=NEUTRAL, linewidth=0.8, linestyle=":")
    top.set(title="Figura 7. Curva de capital neta de costos por partición",
            ylabel="Valor del portafolio (millones de USD)")
    top.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    mark_splits(top, split_starts)

    bottom.fill_between(equity.index, equity["drawdown"] * 100, 0, color="#e34948", alpha=0.35, linewidth=0)
    bottom.plot(equity.index, equity["drawdown"] * 100, color="#e34948", linewidth=0.8)
    bottom.set(ylabel="Drawdown (%)", xlabel="Fecha")
    for name, start in split_starts.items():
        if name != "train":
            bottom.axvline(start, color=NEUTRAL, linestyle="--", linewidth=0.9)
    return save(fig, folder, "fig07_curvas_capital.png")


def plot_trade_returns(trades: pd.DataFrame, folder: Path) -> Path:
    """Figura 8: distribución de retornos netos por operación con una normal superpuesta."""
    returns = trades["return"] * 100
    mu, sigma = returns.mean(), returns.std()
    fig, ax = plt.subplots()
    bins = np.linspace(returns.min(), returns.max(), 13)
    stacked = [returns[trades["split"] == name] for name in SPLIT_NAMES]   # barras apiladas por partición
    ax.hist(stacked, bins=bins, stacked=True, density=True, color=[SPLIT_COLORS[n] for n in SPLIT_NAMES],
            edgecolor="white", linewidth=0.8, label=[f"{SPLIT_NAMES[n]} ({len(s)})" for n, s in zip(SPLIT_NAMES, stacked)])
    x = np.linspace(returns.min() - sigma, returns.max() + sigma, 300)
    ax.plot(x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), color="#2b2b2b",
            label=f"Normal (media {mu:.2f}%, desv. {sigma:.2f}%)")
    ax.axvline(0, color=NEUTRAL, linewidth=0.9, linestyle=":")
    ax.set(title=f"Figura 8. Distribución de retornos netos por operación (n = {len(returns)})",
           xlabel="Retorno neto por operación (% del valor del portafolio al entrar)", ylabel="Densidad")
    ax.legend(loc="upper left", fontsize=9)
    return save(fig, folder, "fig08_retornos_operacion.png")
