# Plan de trabajo y división de tareas — Lab03_MyST_Equipo7

## Principio de reparto

Cada integrante es dueño de una mitad del pipeline y la explica en la exposición:

- **Jerónimo Rojas (@JeroAlva7): del dato a la señal.** Datos, selección por cointegración, filtros de Kalman 1 y 2,
  Johansen móvil. Responde preguntas sobre cointegración, Kalman (Q, R, ganancia, causalidad) y la formulación de Powell
  de cada filtro.
- **Raúl Zanatta (@rzanatta0): de la señal al dinero.** VECM, reglas de operación, sizing, backtest con costos,
  walk-forward, métricas y dashboard. Responde preguntas sobre la lógica de trading, la contabilidad multi-pata, los
  costos y los resultados, y sobre la formulación de Powell del sistema completo.

Los dos deben poder explicar cualquier línea (sección 5 de los lineamientos): cada uno revisa los commits del otro.

## Commits planeados

Un commit por cada pieza terminada y probada. Mensajes con prefijo: `feat`, `fix`, `test`, `docs`, `build`, `chore`.

### Jerónimo — del dato a la señal

| Día | # | Mensaje del commit |
|---|---|---|
| mié 4 | J1 | `docs: plan de trabajo y división de tareas` |
| mié 4 | J2 | `feat(data): descarga de precios ajustados y guardado en CSV` |
| mié 4 | J3 | `feat(data): validación de datos y partición 60/20/20` |
| mié 4 | J4 | `feat(cointegration): verificación de orden de integración I(1) con ADF` |
| mié 4 | J5 | `feat(cointegration): correlación móvil y filtro rho > 0.7` |
| mié 4 | J6 | `feat(cointegration): pre-filtro Engle-Granger por subgrupo` |
| mié 4 | J7 | `feat(cointegration): Johansen con estadísticos de traza y máximo eigenvalor` |
| mié 4 | J8 | `feat(cointegration): validación del spread, vida media y tabla de grupos probados` |
| jue 5 | J9 | `feat(kalman): filtro 1 de pesos dinámicos con normalización` |
| jue 5 | J10 | `feat(kalman): filtro 2 local-level con z-score prior` |
| jue 5 | J11 | `feat(cointegration): Johansen móvil del primer eigenvector` |
| jue 5 | J12 | `test: causalidad del z-score` |
| jue 5 | J13 | `test: normalización del vector de pesos` |
| jue 5 | J14 | `test: innovaciones normalizadas del filtro` |
| vie 6 | J15 | `feat(figures): figuras 1, 3, 4, 5 y 6` |
| vie 6 | J16 | `docs: ejemplo numérico de Kalman y formulación de Powell de ambos filtros` |
| sáb 7 | J17 | `docs: supuestos del modelo en README` |
| dom 8 | J18 | `docs: secciones de datos, cointegración y filtros del reporte` |

### Raúl — de la señal al dinero

| Día | # | Mensaje del commit |
|---|---|---|
| mié 4 | R1 | `docs: corrige enlace de GitHub en README` |
| mié 4 | R2 | `feat(strategy): estimación del VECM con el vector de Johansen` |
| mié 4 | R3 | `feat(strategy): reglas de entrada, salida, stop y tenencia máxima` |
| mié 4 | R4 | `feat(strategy): sizing proporcional a abs(beta) con 80% del capital` |
| jue 5 | R5 | `feat(backtest): motor multi-pata con efectivo y posiciones` |
| jue 5 | R6 | `feat(backtest): comisiones por pata y devengo diario del préstamo` |
| jue 5 | R7 | `feat(backtest): bitácora de operaciones` |
| jue 5 | R8 | `test: contabilidad multi-pata` |
| vie 6 | R9 | `feat(backtest): grid search de umbrales sobre entrenamiento` |
| vie 6 | R10 | `feat(backtest): evaluación walk-forward en prueba y validación` |
| vie 6 | R11 | `feat(backtest): rebalanceo diario contra banda de no-rebalanceo` |
| vie 6 | R12 | `feat(performance): métricas de desempeño y Sharpe móvil de 252 días` |
| vie 6 | R13 | `feat(main): integración del pipeline y escritura de artifacts` |
| sáb 7 | R14 | `feat(dashboard): vista ejecutiva` |
| sáb 7 | R15 | `feat(dashboard): vista técnica` |
| sáb 7 | R16 | `feat(figures): figuras 2, 7 y 8` |
| sáb 7 | R17 | `docs: guía de uso del dashboard en README` |
| dom 8 | R18 | `docs: secciones de estrategia, backtest y resultados del reporte` |

### Compartido (domingo 8)

| Entregable | Quién lo sube |
|---|---|
| `docs/reporte.pdf` (versión final) | Jerónimo |
| Formulación de Powell del sistema completo (reporte) | Raúl |
| `project_management/postmortem.md` | Cada uno su sección, un commit cada quien |
| Estado final de `sprint_01.md` | Cada uno marca sus tareas |
| Sección "Uso de herramientas de IA" del README | Jerónimo |
| Presentación (10 min, 5 cada uno) | Cada uno sus diapositivas |

**Meta:** último commit antes de las 20:00 del domingo 8.

## Reglas de trabajo con Git

1. **Al empezar:** `git pull`. Siempre.
2. **Al terminar cada pieza:** probarla, hacer commit y push de inmediato. No acumular.
3. Cada quien edita los archivos de su mitad. Los archivos compartidos (`main.py`, `README.md`, `config.py`) se editan
   solo después de hacer `git pull`, y se suben en cuanto se terminan para no generar conflictos.
4. El correo de `git config user.email` debe ser el de la cuenta de GitHub de cada uno; si no, los commits no cuentan
   como contribución.

## Contrato entre las dos mitades

Raúl puede avanzar sin esperar a Jerónimo si respeta estos nombres. Mientras no estén listos los datos reales, prueba
sus módulos con un spread sintético (por ejemplo, un proceso AR(1) generado con la semilla de `config.py`).

| Archivo en `artifacts/` | Lo escribe | Columnas (índice: fecha) |
|---|---|---|
| `prices.csv` | data | un precio ajustado por activo |
| `selection_table.csv` | cointegration | grupo, etapa de descarte, estadísticos, vida media |
| `johansen_vector.json` | cointegration | activos, vector beta normalizado, eigenvalores |
| `filter1.csv` | kalman | `beta_<ticker>`, `spread`, `gain_<ticker>`, `var_<ticker>` |
| `filter2.csv` | kalman | `mu_prior`, `sigma_prior`, `z`, `gain`, `var_post` |
| `johansen_rolling.csv` | cointegration | `evec_<ticker>` |
| `signals.csv` | strategy | `z`, `direction` (−1, 0, +1) |
| `equity.csv` | backtest | `equity`, `cash`, `split` (train/test/val), `drawdown` |
| `trades.csv` | backtest | entrada, salida, dirección, P&L bruto, comisiones, préstamo, P&L neto |
| `metrics.json` | performance | métricas por partición |

El dashboard solo lee estos archivos; nunca vuelve a ejecutar el backtest.
