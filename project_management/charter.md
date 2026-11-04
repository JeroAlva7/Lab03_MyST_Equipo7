# Project Charter — Lab03_MyST_Equipo7

## Declaración de alcance

**La estrategia hará:**
- Seleccionar un grupo de 4 activos con justificación económica y cointegración verificada (Johansen) solo con datos de entrenamiento.
- Estimar pesos dinámicos con el filtro de Kalman 1 y un z-score causal con el filtro de Kalman 2.
- Generar señales con un VECM y umbrales calibrados por grid search sobre entrenamiento.
- Evaluar con backtest diario que incluye comisiones (0.125% por pata) y préstamo en cortos (0.25% anual).
- Presentar resultados en un dashboard de Streamlit con vista ejecutiva y técnica.

**La estrategia no hará:**
- Risk Parity entre varios grupos (sección 3.7, omitida por indicación del profesor).
- Operación intradía, ejecución en vivo ni modelado de slippage o impacto de mercado.
- Optimización de parámetros con datos de prueba o validación.

## Plan de hitos

| Hito | Entregable | Fecha objetivo |
|---|---|---|
| H1 | Repositorio, entorno y charter | mar 3 nov |
| H2 | Pipeline de datos completo | mié 4 nov |
| H3 | Prueba de cointegración completa y grupo seleccionado | mié 4 nov |
| H4 | Filtro 1 validado (y filtro 2) | jue 5 nov |
| H5 | VECM, señales y backtest completo con costos | vie 6 nov |
| H6 | Dashboard funcionando y figuras obligatorias | sáb 7 nov |
| H7 | Reporte en docs/reporte.pdf y presentación | dom 8 nov (último commit antes de 20:00) |

## Registro de riesgos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| R1 | Pérdida de cointegración fuera de muestra | Alta | Alto | Stop en \|z\|, tenencia máxima ligada a la vida media, reporte honesto por partición |
| R2 | Divergencia o mala calibración del filtro de Kalman | Media | Alto | Burn-in, diagnóstico de innovaciones normalizadas (prueba 4), Q y R justificados |
| R3 | Problemas de calidad de datos (huecos, splits, cambios de API) | Media | Medio | Precios ajustados, validación automática, datos guardados en CSV dentro del repo |
| R4 | Sesgo de look-ahead | Media | Alto | Z-score con estadísticos prior, Johansen solo en entrenamiento, prueba de causalidad |
| R5 | Pruebas múltiples: relación espuria | Media | Alto | Universo acotado (11 grupos) con racional económico; se reporta el número de grupos probados |
| R6 | Plazo de una semana y dependencia entre integrantes | Media | Medio | Hitos diarios, módulos con interfaces claras, commits frecuentes de ambos |

## Asignación de roles

| Integrante | Responsabilidades | Contribuciones esperadas en Git |
|---|---|---|
| Jerónimo Rojas (@JeroAlva7) | Datos, cointegración, filtros de Kalman, pruebas | `data/`, `cointegration/`, `kalman/`, `tests/` |
| _[Integrante 2]_ | VECM y señales, backtest, métricas, dashboard | `strategy/`, `backtest/`, `performance/`, `dashboard/` |
| Ambos | Reporte, presentación, revisión cruzada | `docs/`, `project_management/`, `README.md` |
