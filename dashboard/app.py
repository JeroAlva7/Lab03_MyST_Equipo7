"""Dashboard del Laboratorio 03 (Streamlit + Plotly). Se ejecuta con:

    streamlit run dashboard/app.py

Lee únicamente los archivos de artifacts/ que escribe `python main.py`; nunca vuelve a correr
el backtest. Tiene dos vistas (sección 3.8 de los lineamientos):
- Ejecutiva, para un portfolio manager: capital, tarjeta de desempeño, costos, bitácora y
  distribución de retornos.
- Técnica, para un investigador cuantitativo: spread, z-score, pesos, eigenvector móvil,
  ganancia de Kalman, varianza posterior y Sharpe móvil.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402  (requiere el path de arriba)
from reporting.figures import ASSET_COLORS, NEUTRAL, SPLIT_NAMES  # noqa: E402

SPLIT_COLORS = {"train": "#2a78d6", "test": "#eb6834", "val": "#1baf7a"}
PERIODS = {"total": "Muestra completa", **SPLIT_NAMES}
ENTRY_EVENTS = {"entrada larga": ("triangle-up", "#1baf7a"), "entrada corta": ("triangle-down", "#e34948")}
EXIT_EVENTS = {"salida": ("Salida por reversión", "#6b6b6b"), "stop": ("Stop", "#e34948"),
               "tiempo": ("Salida por tiempo", "#eda100")}


# --------------------------------------------------------------------------- datos
@st.cache_data
def load_artifacts() -> dict:
    """Lee las salidas del pipeline (se cachean: el dashboard nunca recalcula el backtest)."""
    out = config.ARTIFACTS_DIR

    def csv(name: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(out / name, **kwargs)

    def dated(name: str) -> pd.DataFrame:
        return csv(name, index_col=0, parse_dates=True)

    with open(out / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(out / "kalman_params.json", encoding="utf-8") as f:
        kalman = json.load(f)
    return {"equity": dated("equity.csv"), "signals": dated("signals.csv"),
            "filter1": dated("filter1.csv"), "filter2": dated("filter2.csv"),
            "rolling": dated("johansen_rolling.csv"), "rebalancing": csv("rebalancing.csv"),
            "trades": csv("trades.csv", parse_dates=["entry_date", "exit_date"]),
            "metrics": metrics, "assets": kalman["assets_order"]}


def split_starts(equity: pd.DataFrame) -> dict:
    """Primera fecha de cada partición, para marcarlas en las gráficas."""
    return {name: equity.index[(equity["split"] == name).to_numpy()][0] for name in SPLIT_NAMES}


def cost_attribution(trades: pd.DataFrame) -> dict:
    """P&L bruto descompuesto en comisiones, préstamo y P&L neto (suma de la bitácora)."""
    return {"gross": float(trades["gross_pnl"].sum()), "commission": float(trades["commission"].sum()),
            "borrow": float(trades["borrow_cost"].sum()), "net": float(trades["net_pnl"].sum())}


def filter_trades(trades: pd.DataFrame, splits: list, directions: list, results: list,
                  reasons: list) -> pd.DataFrame:
    """Operaciones que cumplen los filtros de la bitácora."""
    keep = (trades["split"].isin(splits) & trades["direction"].isin(directions)
            & trades["result"].isin(results) & trades["exit_reason"].isin(reasons))
    return trades[keep]


def normal_curve(returns: pd.Series, points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Densidad normal con la media y la desviación de los retornos por operación."""
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min() - sigma, returns.max() + sigma, points)
    return x, np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# --------------------------------------------------------------------------- gráficas
def mark_splits(fig: go.Figure, starts: dict) -> None:
    """Línea vertical al inicio de prueba y de validación, en todos los paneles."""
    for name, start in starts.items():
        if name != "train":
            fig.add_vline(x=start, line_dash="dash", line_color=NEUTRAL, line_width=1)
            fig.add_annotation(x=start, y=1.0, yref="paper", text=SPLIT_NAMES[name], showarrow=False,
                               xanchor="left", yanchor="top", xshift=3,
                               font={"size": 11, "color": NEUTRAL}, bgcolor="rgba(255,255,255,0.7)")


def style(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Formato común: título, leyenda horizontal, zoom con selector de rango y hover unificado."""
    fig.update_layout(title=title, height=height, hovermode="x unified", margin={"t": 70, "b": 40},
                      legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0})
    return fig


def fig_equity(equity: pd.DataFrame, starts: dict) -> go.Figure:
    """Curva de capital por partición con el drawdown debajo."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    for name, label in SPLIT_NAMES.items():
        mask = (equity["split"] == name).to_numpy(copy=True)   # copia editable (pandas 3)
        first = max(mask.argmax() - 1, 0)                  # une el tramo con el anterior
        mask[first] = True
        fig.add_trace(go.Scatter(x=equity.index[mask], y=equity["equity"][mask], name=label,
                                 line={"color": SPLIT_COLORS[name], "width": 1.6}), row=1, col=1)
    fig.add_trace(go.Scatter(x=equity.index, y=equity["drawdown"] * 100, name="Drawdown",
                             fill="tozeroy", line={"color": "#e34948", "width": 1}), row=2, col=1)
    fig.update_yaxes(title_text="Valor del portafolio (USD)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)
    mark_splits(fig, starts)
    return style(fig, "Curva de capital neta de costos y drawdown", height=560)


def fig_costs(attribution: dict) -> go.Figure:
    """Cascada: P&L bruto -> comisiones -> préstamo -> P&L neto."""
    fig = go.Figure(go.Waterfall(
        x=["P&L bruto", "Comisiones", "Préstamo", "P&L neto"],
        y=[attribution["gross"], -attribution["commission"], -attribution["borrow"], attribution["net"]],
        measure=["absolute", "relative", "relative", "total"],
        text=[f"${v:,.0f}" for v in (attribution["gross"], -attribution["commission"],
                                    -attribution["borrow"], attribution["net"])],
        textposition="outside", connector={"line": {"color": NEUTRAL}},
        increasing={"marker": {"color": "#1baf7a"}}, decreasing={"marker": {"color": "#e34948"}},
        totals={"marker": {"color": "#2a78d6"}}))
    fig.update_yaxes(title_text="USD", tickformat="$,.0f")
    fig.update_layout(hovermode="closest")
    return style(fig, "Atribución de costos", height=420)


def fig_return_hist(trades: pd.DataFrame) -> go.Figure:
    """Histograma de retornos por operación con la normal de misma media y desviación."""
    returns = trades["return"] * 100
    fig = go.Figure(go.Histogram(x=returns, histnorm="probability density", name="Operaciones",
                                 marker={"color": "#2a78d6", "line": {"color": "white", "width": 1}},
                                 nbinsx=max(5, min(20, len(returns)))))
    if len(returns) > 2 and returns.std() > 0:
        x, pdf = normal_curve(returns)
        fig.add_trace(go.Scatter(x=x, y=pdf, name="Normal ajustada", line={"color": "#e34948"}))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_xaxes(title_text="Retorno neto por operación (% del valor del portafolio)", ticksuffix="%")
    fig.update_yaxes(title_text="Densidad")
    fig.update_layout(hovermode="closest", bargap=0.02)
    return style(fig, f"Distribución de retornos por operación (n = {len(returns)})")


def fig_spread(spread: pd.Series, signals: pd.DataFrame, starts: dict) -> go.Figure:
    """Spread S_t del filtro 1 con las señales de entrada y salida."""
    fig = go.Figure(go.Scatter(x=spread.index, y=spread, name="Spread Sₜ", line={"color": NEUTRAL, "width": 1}))
    for event, (symbol, color) in ENTRY_EVENTS.items():
        days = signals.index[signals["event"] == event]
        fig.add_trace(go.Scatter(x=days, y=spread[days], mode="markers", name=event.capitalize(),
                                 marker={"symbol": symbol, "size": 11, "color": color}))
    for event, (label, color) in EXIT_EVENTS.items():
        days = signals.index[signals["event"] == event]
        fig.add_trace(go.Scatter(x=days, y=spread[days], mode="markers", name=label,
                                 marker={"symbol": "x", "size": 9, "color": color}))
    fig.update_yaxes(title_text="Spread (log-precios)")
    mark_splits(fig, starts)
    return style(fig, "Spread del portafolio con señales (fecha de decisión; se ejecutan al día siguiente)")


def fig_zscore(signals: pd.DataFrame, params: dict, starts: dict) -> go.Figure:
    """z-score del filtro 2 con los umbrales elegidos en entrenamiento."""
    fig = go.Figure(go.Scatter(x=signals.index, y=signals["z"], name="z-score", line={"color": "#2a78d6", "width": 1}))
    lines = [(params["z_entry"], "Entrada", "#1baf7a", "dash"), (params["z_stop"], "Stop", "#e34948", "dot")]
    for level, label, color, dash in lines:
        for sign in (1, -1):
            fig.add_hline(y=sign * level, line_color=color, line_dash=dash, line_width=1,
                          annotation_text=f"{label} {sign * level:+.1f}" if sign > 0 else None)
    fig.add_hline(y=params["z_exit"], line_color=NEUTRAL, line_width=1,
                  annotation_text=f"Salida {params['z_exit']:.1f}")
    fig.update_yaxes(title_text="z-score")
    mark_splits(fig, starts)
    return style(fig, "z-score con umbrales de entrada, salida y stop")


def fig_weights(filter1: pd.DataFrame, assets: list, starts: dict) -> go.Figure:
    """Pesos beta_t del filtro 1, una línea por activo."""
    fig = go.Figure()
    for asset in assets:
        fig.add_trace(go.Scatter(x=filter1.index, y=filter1[f"beta_{asset}"], name=asset,
                                 line={"color": ASSET_COLORS.get(asset), "width": 1.4}))
    fig.update_yaxes(title_text="Peso en el vector (DUK fijo en +1)")
    mark_splits(fig, starts)
    return style(fig, "Vector de pesos β̂ₜ (filtro de Kalman 1)")


def fig_eigenvector(rolling: pd.DataFrame, assets: list, starts: dict) -> go.Figure:
    """Primer eigenvector de Johansen móvil (norma 1, DUK positivo) y si la ventana cointegra."""
    fig = go.Figure()
    for asset in assets:
        fig.add_trace(go.Scatter(x=rolling.index, y=rolling[f"evec_{asset}"], name=asset, mode="lines+markers",
                                 line={"color": ASSET_COLORS.get(asset), "width": 1.2}, marker={"size": 4}))
    coint = rolling.index[rolling["cointegrated"].astype(bool)]
    fig.add_trace(go.Scatter(x=coint, y=np.full(len(coint), -1.05), mode="markers", name="Ventana cointegrada",
                             marker={"symbol": "square", "size": 6, "color": "#1baf7a"}))
    fig.update_yaxes(title_text="Componente del eigenvector", range=[-1.1, 1.05])
    mark_splits(fig, starts)
    return style(fig, f"Primer eigenvector de Johansen móvil (ventanas de {config.ROLLING['window']} días)")


def fig_gain(filter1: pd.DataFrame, filter2: pd.DataFrame, assets: list, starts: dict) -> go.Figure:
    """Ganancia de Kalman de ambos filtros: con qué agresividad se actualiza cada estimación."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Filtro 1: |ganancia| de cada peso", "Filtro 2: ganancia de la media μₜ"))
    for asset in assets[1:]:
        fig.add_trace(go.Scatter(x=filter1.index, y=filter1[f"gain_{asset}"].abs(), name=f"|K| {asset}",
                                 line={"color": ASSET_COLORS.get(asset), "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=filter2.index, y=filter2["gain"], name="K de μₜ",
                             line={"color": "#2a78d6", "width": 1.2}), row=2, col=1)
    fig.update_yaxes(type="log", dtick=1, exponentformat="power", row=2, col=1)
    mark_splits(fig, starts)
    return style(fig, "Ganancia de Kalman en el tiempo", height=560)


def fig_posterior_variance(filter1: pd.DataFrame, filter2: pd.DataFrame, assets: list, starts: dict) -> go.Figure:
    """Varianza posterior de cada estado: la incertidumbre del filtro y su convergencia."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Filtro 1: varianza posterior de cada peso", "Filtro 2: varianza posterior de μₜ"))
    for asset in assets[1:]:
        fig.add_trace(go.Scatter(x=filter1.index, y=filter1[f"var_{asset}"], name=f"P {asset}",
                                 line={"color": ASSET_COLORS.get(asset), "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=filter2.index, y=filter2["var_post"], name="P de μₜ",
                             line={"color": "#2a78d6", "width": 1.2}), row=2, col=1)
    fig.update_yaxes(type="log", dtick=1, exponentformat="power", row=1, col=1)
    fig.update_yaxes(type="log", dtick=1, exponentformat="power", row=2, col=1)
    mark_splits(fig, starts)
    return style(fig, "Diagnóstico de convergencia: varianza posterior (escala log)", height=560)


def fig_rolling_sharpe(equity: pd.DataFrame, starts: dict) -> go.Figure:
    """Sharpe anualizado en ventanas móviles de 252 días."""
    fig = go.Figure(go.Scatter(x=equity.index, y=equity["rolling_sharpe"], name="Sharpe móvil",
                               line={"color": "#2a78d6", "width": 1.2}))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_yaxes(title_text="Sharpe anualizado (252 días)")
    mark_splits(fig, starts)
    return style(fig, "Sharpe móvil de 252 días")


# --------------------------------------------------------------------------- vistas
def fmt_ratio(value: float) -> str:
    """Número con dos decimales; infinito para un profit factor sin pérdidas."""
    return "∞" if np.isinf(value) else f"{value:.2f}"


def performance_card(card: dict) -> None:
    """Tarjeta de desempeño de la partición elegida."""
    top = st.columns(6)
    top[0].metric("Sharpe", fmt_ratio(card["sharpe"]))
    top[1].metric("Sortino", fmt_ratio(card["sortino"]))
    top[2].metric("Calmar", fmt_ratio(card["calmar"]))
    top[3].metric("Max drawdown", f"{card['max_drawdown']:.1%}")
    top[4].metric("Win rate", f"{card['win_rate']:.0%}")
    top[5].metric("Profit factor", fmt_ratio(card["profit_factor"]))
    bottom = st.columns(6)
    bottom[0].metric("Rendimiento", f"{card['return']:.1%}")
    bottom[1].metric("Rend. anual", f"{card['annual_return']:.2%}")
    bottom[2].metric("Volatilidad anual", f"{card['annual_volatility']:.2%}")
    bottom[3].metric("P&L neto", f"${card['net_pnl']:,.0f}")
    bottom[4].metric("Operaciones", card["trades"])
    bottom[5].metric("Periodo", f"{card['start'][:4]}–{card['end'][:4]}",
                     help=f"Del {card['start']} al {card['end']}")


def trade_log_view(trades: pd.DataFrame) -> pd.DataFrame:
    """Filtros de la bitácora y tabla; regresa las operaciones filtradas."""
    cols = st.columns(4)
    splits = cols[0].multiselect("Periodo", list(SPLIT_NAMES), default=list(SPLIT_NAMES),
                                 format_func=SPLIT_NAMES.get)
    directions = cols[1].multiselect("Dirección del spread", ["larga", "corta"], default=["larga", "corta"])
    results = cols[2].multiselect("Resultado", ["ganadora", "perdedora"], default=["ganadora", "perdedora"])
    reasons = sorted(trades["exit_reason"].unique())
    chosen_reasons = cols[3].multiselect("Motivo de salida", reasons, default=reasons)
    shown = filter_trades(trades, splits, directions, results, chosen_reasons)

    table = shown.assign(split=shown["split"].map(SPLIT_NAMES), ret=shown["return"] * 100)
    st.dataframe(
        table[["entry_date", "exit_date", "split", "direction", "days", "exit_reason",
               "gross_pnl", "commission", "borrow_cost", "net_pnl", "ret", "result"]],
        hide_index=True,
        column_config={
            "entry_date": st.column_config.DateColumn("Entrada", format="YYYY-MM-DD"),
            "exit_date": st.column_config.DateColumn("Salida", format="YYYY-MM-DD"),
            "split": "Periodo", "direction": "Dirección", "days": "Días", "exit_reason": "Motivo",
            "gross_pnl": st.column_config.NumberColumn("P&L bruto", format="dollar"),
            "commission": st.column_config.NumberColumn("Comisiones", format="dollar"),
            "borrow_cost": st.column_config.NumberColumn("Préstamo", format="dollar"),
            "net_pnl": st.column_config.NumberColumn("P&L neto", format="dollar"),
            "ret": st.column_config.NumberColumn("Retorno", format="%.2f%%"),
            "result": "Resultado"})
    st.caption(f"{len(shown)} de {len(trades)} operaciones · P&L neto filtrado: ${shown['net_pnl'].sum():,.0f}")
    return shown


def executive_view(data: dict, starts: dict) -> None:
    """Vista para el portfolio manager o el comité de inversión."""
    period = st.selectbox("Periodo de la tarjeta y de la atribución de costos", list(PERIODS),
                          format_func=PERIODS.get)
    performance_card(data["metrics"]["by_split"][period])
    st.plotly_chart(fig_equity(data["equity"], starts), key="equity")

    trades = data["trades"]
    period_trades = trades if period == "total" else trades[trades["split"] == period]
    attribution = cost_attribution(period_trades)
    left, right = st.columns([3, 2])
    left.plotly_chart(fig_costs(attribution), key="costs")
    share = attribution["commission"] / attribution["gross"] if attribution["gross"] > 0 else float("nan")
    right.markdown(f"**{PERIODS[period]}**")
    right.markdown(f"- Comisiones: **${attribution['commission']:,.0f}** "
                   f"({share:.0%} del P&L bruto)\n"
                   f"- Préstamo de los cortos: **${attribution['borrow']:,.0f}**\n"
                   f"- Comisión: {config.COMMISSION:.3%} por pata en entradas, salidas y rebalanceos; "
                   f"préstamo: {config.BORROW_RATE_ANNUAL:.2%} anual sobre el nocional corto.")

    st.subheader("Bitácora de operaciones")
    shown = trade_log_view(trades)
    if len(shown):
        st.plotly_chart(fig_return_hist(shown), key="returns")


def technical_view(data: dict, starts: dict) -> None:
    """Vista para el investigador cuantitativo o el gestor de riesgos."""
    params, assets = data["metrics"]["params"], data["assets"]
    signals, filter1, filter2 = data["signals"], data["filter1"], data["filter2"]
    st.plotly_chart(fig_spread(filter1["spread"], signals, starts), key="spread")
    st.plotly_chart(fig_zscore(signals, params, starts), key="zscore")
    st.plotly_chart(fig_weights(filter1, assets, starts), key="weights")
    st.plotly_chart(fig_eigenvector(data["rolling"], assets, starts), key="eigenvector")
    st.plotly_chart(fig_gain(filter1, filter2, assets, starts), key="gain")
    st.plotly_chart(fig_posterior_variance(filter1, filter2, assets, starts), key="variance")
    st.plotly_chart(fig_rolling_sharpe(data["equity"], starts), key="rolling_sharpe")

    with st.expander("Costo del rebalanceo diario contra bandas de no-rebalanceo"):
        reb = data["rebalancing"].set_index("policy")
        st.dataframe(reb[["rebalance_days", "commission_rebalance", "net_pnl", "sharpe",
                          "exposure_min", "exposure_max"]],
                     column_config={
                         "rebalance_days": "Días con rebalanceo",
                         "commission_rebalance": st.column_config.NumberColumn("Comisión de rebalanceo", format="dollar"),
                         "net_pnl": st.column_config.NumberColumn("P&L neto", format="dollar"),
                         "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                         "exposure_min": st.column_config.NumberColumn("Exposición mínima", format="percent"),
                         "exposure_max": st.column_config.NumberColumn("Exposición máxima", format="percent")})


def sidebar(data: dict) -> None:
    """Resumen de la estrategia y de los parámetros elegidos en entrenamiento."""
    params = data["metrics"]["params"]
    st.sidebar.header("Estrategia")
    st.sidebar.markdown(f"**Grupo:** {', '.join(data['assets'])}  \n"
                        f"**Capital inicial:** ${config.INITIAL_CAPITAL:,.0f} · {config.CAPITAL_FRACTION:.0%} desplegado")
    st.sidebar.header("Parámetros (grid search en entrenamiento)")
    st.sidebar.markdown(f"- z de entrada: **±{params['z_entry']:.2f}**\n"
                        f"- z de salida: **{params['z_exit']:.2f}**\n"
                        f"- Stop: **|z| > {params['z_stop']:.1f}**\n"
                        f"- Tenencia máxima: **{params['max_holding_days']} días** "
                        f"({params['max_holding_multiple']:.0f} × vida media de {params['spread_half_life']:.1f})\n"
                        f"- Rebalanceo: **diario**")
    st.sidebar.caption("Los datos vienen de artifacts/ (python main.py). "
                       "Use el zoom, el hover y la leyenda de cada gráfica para explorar.")


def main() -> None:
    """Arma la página con las dos vistas."""
    st.set_page_config(page_title="Lab03 MyST — Equipo 7", layout="wide")
    st.title("Arbitraje estadístico con cointegración y filtros de Kalman")
    try:
        data = load_artifacts()
    except FileNotFoundError as error:
        st.error(f"Falta un archivo de artifacts/ ({error.filename}). Corra primero: python main.py")
        st.stop()
    starts = split_starts(data["equity"])
    sidebar(data)
    executive, technical = st.tabs(["Vista ejecutiva", "Vista técnica"])
    with executive:
        executive_view(data, starts)
    with technical:
        technical_view(data, starts)


if __name__ == "__main__":
    main()
